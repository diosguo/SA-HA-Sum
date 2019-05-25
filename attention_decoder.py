import tensorflow as tf



def attention_decoder(decoder_inputs,
                      initial_state,
                      encoder_states,
                      enc_padding_mask,
                      cell,
                      initial_state_attention=False,
                      pointer_gen=True,
                      use_coverage=False,
                      prev_coverage=None):
    with tf.variable_scope('attention_decoder') as scope:
        batch_size = encoder_states.get_shape()[0].value
        attn_size = encoder_states.get_shape()[2].value

        encoder_states = tf.expand_dims(encoder_states,axis=2)

        attention_vec_size = attn_size

        W_h = tf.get_variable('W_h',[1,1,attn_size, attention_vec_size])

        encoder_features = tf.nn.conv2d(encoder_states,W_h,[1,1,1,1],'SAME')

        v = tf.get_variable('v',[attention_vec_size])
        if use_coverage:
            with tf.variable_scope('coverage'):
                w_c = tf.get_variable('w_c',[1,1,1,attention_vec_size])

        if prev_coverage:
            prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage,2),3)


        def attention(decoder_state, coverage=None):

            with tf.variable_scope('Attention'):
                decoder_features = linear(decoder_state, attention_vec_size, True)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features,1),1)






def linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (isinstance(args,(list,tuple)) and not args):
        raise ValueError("'args' must be specified")
    if not isinstance(args,(list,tuple)):
        args=[args]

    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]

    for shape in shapes:
        if len(shape)!=2:
            raise ValueError('Linear is expecting 2D arguemnts %s'%str(shapes))
        if not shape[1]:
            raise ValueError('Linear is expects shape[1] of arguments:%s'%str(shapes))
        else:
            total_arg_size += shape[1]

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args)==1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, value=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable('Bias',[output_size],initializer=tf.constant_initializer(bias_start))

    return res+bias_term