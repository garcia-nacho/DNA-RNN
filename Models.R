#loss: 1.2801 - categorical_accuracy: 0.3800 - val_loss: 1.2817 - val_categorical_accuracy: 0.3844


InputDNA<-layer_input(shape = (ncol(genome.x))) 

Layers<-InputDNA %>% 
  layer_embedding(input_dim =  5, output_dim =  42 ,
                  mask_zero=FALSE,
                  input_length=input.len, name = "Embd") %>% 
  layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%  
  layer_conv_1d(filters=64, kernel_size=8,  activation='relu', padding='same',  strides=1) %>% 
  layer_conv_1d(filters=96, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
  layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%  
  layer_conv_1d(filters=64, kernel_size=8,  activation='relu', padding='same',  strides=1) %>% 
  layer_conv_1d(filters=96, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
  layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%  
  layer_conv_1d(filters=64, kernel_size=8,  activation='relu', padding='same',  strides=1) %>% 
  layer_conv_1d(filters=96, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
  layer_global_max_pooling_1d() %>% 
  layer_dense(units = 128) %>% 
  layer_dense(units = 28) %>% 
  layer_dense(units = 4, activation = "softmax")

model<-keras_model(inputs = InputDNA, outputs=Layers )


# Model2 ------------------------------------------------------------------


