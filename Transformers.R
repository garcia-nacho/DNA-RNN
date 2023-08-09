transformer_encoder <- function(inputs,
                                head_size,
                                num_heads,
                                ff_dim,
                                dropout = 0) {
  # Attention and Normalization
  attention_layer <-
    layer_multi_head_attention(key_dim = head_size,
                               num_heads = num_heads,
                               dropout = dropout)
  
  n_features <- dim(inputs) %>% tail(1)
  
  x <- inputs %>%
    attention_layer(., .) %>%
    layer_dropout(dropout) %>%
    layer_layer_normalization(epsilon = 1e-6)
  
  res <- x + inputs
  
  # Feed Forward Part
  x <- res %>%
    layer_conv_1d(ff_dim, kernel_size = 1, activation = "relu") %>%
    layer_dropout(dropout) %>%
    layer_conv_1d(n_features, kernel_size = 1) %>%
    layer_layer_normalization(epsilon = 1e-6)
  
  # return output + residual
  x + res
}


build_model <- function(input_shape,
                        head_size,
                        num_heads,
                        ff_dim,
                        num_transformer_blocks,
                        mlp_units,
                        dropout = 0,
                        mlp_dropout = 0) {
  
  inputs <- layer_input(input_shape)
  
  x <- inputs
  for (i in 1:num_transformer_blocks) {
    x <- x %>%
      transformer_encoder(
        head_size = head_size,
        num_heads = num_heads,
        ff_dim = ff_dim,
        dropout = dropout
      )
  }
  
  x <- x %>% 
    layer_global_average_pooling_1d(data_format = "channels_first")
  
  for (dim in mlp_units) {
    x <- x %>%
      layer_dense(dim, activation = "relu") %>%
      layer_dropout(mlp_dropout)
  }
  
  outputs <- x %>% 
    layer_dense(n_classes, activation = "softmax")
  
  keras_model(inputs, outputs)
}
