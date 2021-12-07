import tensorflow as tf
import random
import numpy as np

class MyModel(object):
    """multitask模型"""
    def __init__(self, 
                 embedding_dim, 
                 hidden_dim,
                 vocab_size_char, 
                 vocab_size_bio, 
                 vocab_size_type,
                 O_tag_index,
                 use_crf
                 ):
        
        self.inputs_seq = tf.placeholder(tf.int32, [None, None], name="inputs_seq") # B * S
        self.inputs_seq_len = tf.placeholder(tf.int32, [None], name="inputs_seq_len") # B
       
        self.outputs_seq_bio = tf.placeholder(tf.int32, [None, None], name='outputs_seq_bio') # B * S
        self.outputs_seq_type = tf.placeholder(tf.int32, [None, None], name='outputs_seq_type')  # B * S
        self.num = tf.constant(100,name="num")
        
        # 词嵌入层
        with tf.variable_scope('embedding_layer'):
            embedding_matrix = tf.get_variable("embedding_matrix", [vocab_size_char, embedding_dim], dtype=tf.float32)
            embedded = tf.nn.embedding_lookup(embedding_matrix, self.inputs_seq)

        # 编码层
        with tf.variable_scope('encoder1'):
            cell_fw1 = tf.nn.rnn_cell.LSTMCell(hidden_dim)
            cell_bw1 = tf.nn.rnn_cell.LSTMCell(hidden_dim)
            ((rnn_fw_outputs1, rnn_bw_outputs1), (rnn_fw_final_state, rnn_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw1, 
                cell_bw=cell_bw1, 
                inputs=embedded, 
                sequence_length=self.inputs_seq_len,
                dtype=tf.float32
            )
            rnn_outputs1 = tf.concat([rnn_fw_outputs1, rnn_bw_outputs1],axis=2) # B * S * D(2*hidden_dim)
        
        # bio预测层
        with tf.variable_scope('bio_projection'):
            logits_bio = tf.layers.dense(rnn_outputs1, vocab_size_bio) # B * S * V(BIO类别)
            probs_bio = tf.nn.softmax(logits_bio)
            
            if not use_crf:
                preds_bio = tf.argmax(probs_bio, axis=-1, name="preds_bio") # B * S
            else:
                log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_bio, 
                                                                                      self.outputs_seq_bio, 
                                                                                      self.inputs_seq_len)
                preds_bio, crf_scores = tf.contrib.crf.crf_decode(logits_bio, transition_matrix,self.inputs_seq_len)    

        # 102分类的编码层编码层
        with tf.variable_scope('encoder2'):
            mask = tf.greater_equal(preds_bio, 1)# >=1BIOES (32,50)  TF
            mask= tf.cast(mask,dtype=tf.int32) # (32,50)  0 1 
            #num = tf.reduce_sum(mask,0)
            #mask_1 = tf.equal(mask, 1)   # bies
            #mask_0 =tf.equal(mask, 0)    # o
            class_add = tf.multiply(self.inputs_seq,mask)

            class_embedded = tf.nn.embedding_lookup(embedding_matrix, class_add) #（32，50，300）

            class_embedded = tf.reduce_sum(class_embedded,1) #(32,300)he

            add = tf.expand_dims(class_embedded, 1) #(32,1,300)
            add = tf.div(add, tf.cast(self.inputs_seq_len[0],dtype=tf.float32)) #(32,1,300)。。平均
            add = tf.tile(add,[1,self.inputs_seq_len[0],1]) #(32,50,300)

            class_input = tf.concat([embedded, add],axis=2)# (32,50,600)

            cell_fw2 = tf.nn.rnn_cell.LSTMCell(hidden_dim)
            cell_bw2 = tf.nn.rnn_cell.LSTMCell(hidden_dim)
            ((rnn_fw_outputs2, rnn_bw_outputs2), (rnn_fw_final_state, rnn_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw2, 
                cell_bw=cell_bw2, 
                inputs=class_input, 
                sequence_length=self.inputs_seq_len,
                dtype=tf.float32
            )

            rnn_outputs2 = tf.concat([rnn_fw_outputs2, rnn_bw_outputs2],axis=2) # B * S * D(2*hidden_dim)

        # 类别预测层
        with tf.variable_scope('type_projection'):
            logits_type = tf.layers.dense(rnn_outputs2, vocab_size_type) # B * S * V(attr类别)
            probs_type = tf.nn.softmax(logits_type, -1)
            preds_type = tf.argmax(probs_type, axis=-1, name="preds_type") # B * S

        self.outputs = (preds_bio, preds_type)

        # 损失
        with tf.variable_scope('loss'):
            if not use_crf:
                loss_bio = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_bio, labels=self.outputs_seq_bio) # B * S
                masks_bio = tf.sequence_mask(self.inputs_seq_len, dtype=tf.float32) # B * S
                loss_bio = tf.reduce_sum(loss_bio * masks_bio, axis=-1) / tf.cast(self.inputs_seq_len, tf.float32) # B
            else:
                loss_bio = -log_likelihood / tf.cast(self.inputs_seq_len, tf.float32)
 
            loss_type = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_type, labels=self.outputs_seq_type)  # B * S
            masks_type = tf.cast(tf.not_equal(preds_bio, O_tag_index), tf.float32)  # B * S
            loss_type = tf.reduce_sum(loss_type * masks_type, axis=-1) / (tf.reduce_sum(masks_type, axis=-1) + 1e-5)  # B

            loss = loss_bio + loss_type # B
        
        self.loss = tf.reduce_mean(loss)
            
        with tf.variable_scope('opt'):
            self.train_op = tf.train.AdamOptimizer().minimize(loss)


    
