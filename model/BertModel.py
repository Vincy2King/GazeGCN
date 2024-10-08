class BertModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # print(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.eye_embeddings = EyeEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            pos_tag_ids=None,
            dep_tag_ids=None,
            dist_mat=None,
            eye_dist_mat=None,
            FFDs=None,
            GDs=None,
            GPTs=None,
            TRTs=None,
            nFixs=None,
    ):
        """ Forward pass on the Model.
        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.
        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    torch.long
                )  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output, word_embeddings = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        if self.config.revise_gat != 'org':
            # print('ffd.device:',FFDs)
            if self.config.revise_gat == 'ffd':
                eye_embedding_output, eye_embeddings = self.eye_embeddings(eye_feature=FFDs)
            elif self.config.revise_gat == 'gd':
                eye_embedding_output, eye_embeddings = self.eye_embeddings(eye_feature=GDs)
            elif self.config.revise_gat == 'gpt':
                eye_embedding_output, eye_embeddings = self.eye_embeddings(eye_feature=GPTs)
            elif self.config.revise_gat == 'trt':
                eye_embedding_output, eye_embeddings = self.eye_embeddings(eye_feature=TRTs)
            elif self.config.revise_gat == 'nfix':
                eye_embedding_output, eye_embeddings = self.eye_embeddings(eye_feature=nFixs)
            elif self.config.revise_gat == 'all':
                ffd_eye_embedding_output, ffd_eye_embeddings = self.eye_embeddings(eye_feature=FFDs)
                gd_eye_embedding_output, gd_eye_embeddings = self.eye_embeddings(eye_feature=GDs)
                # print('gpts:',GPTs)
                gpt_eye_embedding_output, gpt_eye_embeddings = self.eye_embeddings(eye_feature=GPTs)
                trt_eye_embedding_output, trt_eye_embeddings = self.eye_embeddings(eye_feature=TRTs)
                nfix_eye_embedding_output, nfix_eye_embeddings = self.eye_embeddings(eye_feature=nFixs)
                new_ffd_output = ffd_eye_embedding_output.clone()
                new_gd_output = gd_eye_embedding_output.clone()
                new_gpt_output = gpt_eye_embedding_output.clone()
                new_trt_output = trt_eye_embedding_output.clone()
                new_nfix_output = nfix_eye_embedding_output.clone()

                new_ffd = ffd_eye_embeddings.clone()
                new_gd = gd_eye_embeddings.clone()
                new_gpt = gpt_eye_embeddings.clone()
                new_trt = trt_eye_embeddings.clone()
                new_nfix = nfix_eye_embeddings.clone()
                eye_embedding_output = new_ffd_output+new_gd_output+new_gpt_output+new_trt_output+new_nfix_output
                eye_embeddings = new_ffd+new_gd+new_gpt+new_trt+new_nfix
                # print('ffd_eye_embedding_loss: ',new_ffd_output.device,new_ffd_output)
                # print('new_ffd:',new_ffd.shape,new_ffd.device)
                # print('eye:',eye_embeddings.device)
                # print('gd_eye_embedding_output:',gd_eye_embedding_output.shape,type(gd_eye_embedding_output))
                # print('gpt_eye_embedding_output:',gpt_eye_embedding_output.shape,type(gpt_eye_embedding_output))
                # print('trt_eye_embedding_output:',trt_eye_embedding_output.shape,type(trt_eye_embedding_output))
                # print('nfix_eye_embedding_output:',nfix_eye_embedding_output.shape,type(nfix_eye_embedding_output))

                # print('eye_embedding_output:',eye_embedding_output.shape,eye_embedding_output)
                # print('eye_embeddings:',eye_embeddings.shape,eye_embeddings)
            elif self.config.revise_gat == 'fuse':
                eye_embedding_output, eye_embeddings = self.eye_embeddings(eye_feature=nFixs)

            # print('+++++++++++++++++++++++')
            # print(embedding_output.shape,word_embeddings.shape)
            # print(eye_embedding_output.shape,eye_embeddings.shape)
            # print('eye_embedding_output:',eye_embedding_output)
            encoder_outputs = self.encoder(
                embedding_output,#+embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                pos_tag_ids=pos_tag_ids,
                dep_tag_ids=dep_tag_ids,
                dist_mat=dist_mat,
                eye_dist_mat=eye_dist_mat,
                word_embeddings=word_embeddings,
                FFDs=FFDs,
                GDs=GDs,
                GPTs=GPTs,
                TRTs=TRTs,
                nFixs=nFixs,
                ffd_embeddings=eye_embeddings,
            )
        elif self.config.revise_gat == 'org':
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                pos_tag_ids=pos_tag_ids,
                dep_tag_ids=dep_tag_ids,
                dist_mat=dist_mat,
                eye_dist_mat=eye_dist_mat,
                word_embeddings=word_embeddings,
                FFDs=FFDs,
                GDs=GDs,
                GPTs=GPTs,
                TRTs=TRTs,
                nFixs=nFixs,
            )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:
                                                      ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
