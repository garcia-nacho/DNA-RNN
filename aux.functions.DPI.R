#Auxiliary Functions
library(keras)
library(tensorflow)

# Prob pooling layer TF1 ------------------------------------------------------

GEP_layer_tf1 <- R6::R6Class("GEP:Layer",
                         
                         inherit = KerasLayer,
                         
                         public = list(
                           
                           output_dim = NULL,
                           kernel = NULL,
                           mode=NULL,
                           m_value=NULL,
                           m_trainable=NULL,
                           m=NULL,
                           
                           initialize = function(mode=0, m_value=1, m_trainable=TRUE) {
                             self$mode <- mode
                             self$m_value <- m_value
                             self$m_trainable <- m_trainable
                           },
                           
                           compute_output_shape = function(input_shape) {
                             list(input_shape[[1]], input_shape[[3]])
                           },
                           
                           build = function(input_shape) {
                             
                             if(self$m_trainable){
                               self$m <- self$add_weight(
                                 name = 'm',
                                 shape = list(input_shape[[2]]/input_shape[[2]],1L),
                                 initializer = initializer_constant(value = self$m_value),
                                 trainable = TRUE)
                             }else{
                               self$m <- self$add_weight(
                                 name = 'm',
                                 shape = list(input_shape[[2]]/input_shape[[2]],1L),
                                 initializer = initializer_constant(value = self$m_value),
                                 trainable = FALSE)
                             }
                             
                           },
                           
                           call = function(x, mask=NULL) {
                             if(self$mode==0){
                               now = tf$transpose(x, perm=list(0L,2L,1L))
                               diff_1 = tf$subtract(now, tf$reduce_max(now, axis=-1L, keep_dims=TRUE))
                               diff = tf$multiply(diff_1, self$m)
                               prob = tf$nn$softmax(diff)
                               expectation = tf$reduce_sum(tf$multiply(now, prob), axis=-1L, keep_dims=FALSE)
                             }else{
                               now = tf$transpose(x, perm=list(0L,2L,1L))
                               now_diff = tf$subtract(now, tf$reduce_mean(now, axis=-1L, keep_dims=TRUE))
                               now_diff_m = tf$multiply(now_diff, self$m)
                               sgn_now = tf$sign(now_diff_m)
                               diff_2 = tf$add(tf$multiply(sgn_now, tf$exp(now_diff_m)), tf$exp(now_diff_m))
                               diff_now = tf$div(diff_2,2)
                               prob = diff_now / tf$reduce_sum(diff_now, axis=-1L, keepdims=TRUE)
                               expectation= tf$reduce_sum(tf$multiply(now, prob), axis=-1L, keep_dims=FALSE)
                             }
                             return(expectation)
                             
                           }
                           
                           
                         )
)


Global_Expectation_Pooling_tf1 <- function(object, name = NULL,  m_value=1, mode=1, m_trainable=TRUE) {
  
  create_layer(GEP_layer_tf1, object, list(
    name = name,
    
    m_value=m_value,
    mode=mode,
    m_trainable=m_trainable
  ))
} 







# Prob pooling layer TF2 --------------------------------------------------


GEP_layer <- R6::R6Class("GEP:Layer",
                         
                         inherit = KerasLayer,
                         
                         public = list(
                           
                           output_dim = NULL,
                           kernel = NULL,
                           mode=NULL,
                           m_value=NULL,
                           m_trainable=NULL,
                           m=NULL,
                           
                           initialize = function(mode=0, m_value=1, m_trainable=FALSE) {
                             self$mode <- mode
                             self$m_value <- m_value
                             self$m_trainable <- m_trainable
                           },
                           
                           compute_output_shape = function(input_shape) {
                             list(input_shape[[1]], input_shape[[3]])
                           },
                           
                           build = function(input_shape) {
                             
                             if(self$m_trainable){
                               self$m <- self$add_weight(
                                 name = 'm',
                                 shape = list(as.integer(input_shape[[2]]/input_shape[[2]]),1L),
                                 initializer = initializer_constant(value = self$m_value),
                                 trainable = TRUE)
                             }else{
                               self$m <- self$add_weight(
                                 name = 'm',
                                 shape = list(as.integer(input_shape[[2]]/input_shape[[2]]),1L),
                                 initializer = initializer_constant(value = self$m_value),
                                 trainable = FALSE)
                             }
                             
                           },
                           
                           call = function(x, mask=NULL) {
                             if(self$mode==0){
                               now = tf$transpose(x, perm=list(0L,2L,1L))
                               diff_1 = tf$subtract(now, tf$compat$v1$reduce_max(now, axis=-1L, keep_dims=TRUE))
                               diff = tf$multiply(diff_1, self$m)
                               prob = tf$nn$softmax(diff)
                               expectation = tf$compat$v1$reduce_sum(tf$multiply(now, prob), axis=-1L, keep_dims=FALSE)
                             }else{
                               now = tf$transpose(x, perm=list(0L,2L,1L))
                               now_diff = tf$subtract(now, tf$compat$v1$reduce_mean(now, axis=-1L, keep_dims=TRUE))
                               now_diff_m = tf$multiply(now_diff, self$m)
                               sgn_now = tf$sign(now_diff_m)
                               diff_2 = tf$add(tf$multiply(sgn_now, tf$exp(now_diff_m)), tf$exp(now_diff_m))
                               diff_now = tf$compat$v1$div(diff_2,2)
                               prob = diff_now / tf$compat$v1$reduce_sum(diff_now, axis=-1L, keepdims=TRUE)
                               expectation= tf$compat$v1$reduce_sum(tf$multiply(now, prob), axis=-1L, keep_dims=FALSE)
                             }
                             return(expectation)
                             
                           }
                           
                           
                         )
)


Global_Expectation_Pooling <- function(object, name = NULL,  m_value=1, mode=1, m_trainable=TRUE) {
  
  create_layer(GEP_layer, object, list(
    name = name,
    
    m_value=m_value,
    mode=mode,
    m_trainable=m_trainable
  ))
} 






# DeepDTA model generator -------------------------------------------------

DeepDTA<-function(drug.pad, prot.pad, act="linear", dimension.prot=21, dimension.drug=31, class=FALSE){

InputProt <- layer_input(shape = prot.pad) 
InputSmi <- layer_input(shape = drug.pad)

Prot.Layer<-InputProt %>% 
  layer_embedding(input_dim =  dimension.prot, output_dim =  128 ,
                  mask_zero=TRUE,
                  input_length=1000, name = "LS") %>% 
  layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%  
  layer_conv_1d(filters=64, kernel_size=8,  activation='relu', padding='same',  strides=1) %>% 
  layer_conv_1d(filters=96, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
  layer_global_max_pooling_1d() 

Smi.Layer<-InputSmi %>% 
  layer_embedding(input_dim =  dimension.drug, output_dim =  128 ,
                  mask_zero = TRUE,
                  input_length=100) %>%  
  layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>% 
  layer_conv_1d(filters=64, kernel_size=6,  activation='relu', padding='same',  strides=1) %>% 
  layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1) %>%  
  layer_global_max_pooling_1d()


Prot.Smi<-layer_concatenate(c(Prot.Layer, Smi.Layer), axis = -1) %>% 
  #New hyperparams defining the number of neurons
  layer_dense(1024,  activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(1024, activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(512, activation = "relu") 
  if(class==FALSE){
  Prot.Smi<- Prot.Smi %>% layer_dense(1, kernel_initializer = "normal", activation =act)  
  }else{
  Prot.Smi<- Prot.Smi %>%layer_dense(2,  activation ="softmax")
}
model<-keras_model(inputs = c(InputProt,InputSmi), outputs=Prot.Smi )
return(model)
}



# CNNTXT ------------------------------------------------------------------

CNNTXT<-function(drug.pad, prot.pad){
  InputProt<- layer_input(shape = prot.pad)
  InputSmi<- layer_input(shape = 100)

  Prot.Layer1<-InputProt %>% 
    layer_embedding(input_dim =  21, output_dim =  128,mask_zero=TRUE,input_length=prot.pad) %>% 
    layer_conv_1d(filters=32, kernel_size= 4,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=64, kernel_size= 4,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=96, kernel_size= 4,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  
  Prot.Layer2<-InputProt %>% 
    layer_embedding(input_dim =  21, output_dim =  128,mask_zero=TRUE,input_length=prot.pad) %>% 
    layer_conv_1d(filters=32, kernel_size= 8,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=64, kernel_size= 8,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=96, kernel_size= 8,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Prot.Layer3<-InputProt %>% 
    layer_embedding(input_dim =  21, output_dim =  128,mask_zero=TRUE,input_length=prot.pad) %>% 
    layer_conv_1d(filters=32, kernel_size= 12,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=64, kernel_size= 12,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=96, kernel_size= 12,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  #9th June Ingvild changed input_dim to = 31. 
  Smi.Layer1 <-InputSmi %>% 
    layer_embedding(input_dim =  31, output_dim =  128,mask_zero=TRUE,input_length=drug.pad) %>% 
    layer_conv_1d(filters=32, kernel_size= 4,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(3) %>%
    layer_conv_1d(filters=64, kernel_size= 4,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(3) %>%
    layer_conv_1d(filters=96, kernel_size= 4,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Smi.Layer2 <-InputSmi %>% 
    layer_embedding(input_dim =  31, output_dim =  128,mask_zero=TRUE,input_length=drug.pad) %>% 
    layer_conv_1d(filters=32, kernel_size= 6,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(3) %>%
    layer_conv_1d(filters=64, kernel_size= 6,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(3) %>%
    layer_conv_1d(filters=96, kernel_size= 6,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Smi.Layer3 <-InputSmi %>% 
    layer_embedding(input_dim =  31, output_dim =  128,mask_zero=TRUE,input_length=drug.pad) %>% 
    layer_conv_1d(filters=32, kernel_size= 8,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(3) %>%
    layer_conv_1d(filters=64, kernel_size= 8,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(3) %>%
    layer_conv_1d(filters=96, kernel_size= 8,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 

  
  Prot.combined<- layer_concatenate(c(Prot.Layer1, Prot.Layer2, Prot.Layer3), axis = -1) %>% 
    layer_dense(512,  activation = "relu") %>%
    layer_dropout(0.1) 
  
  Smi.combined<- layer_concatenate(c(Smi.Layer1, Smi.Layer2, Smi.Layer3), axis = -1) %>% 
    layer_dense(512,  activation = "relu") %>%
    layer_dropout(0.1) 
  
  Prot.Smi<-layer_concatenate(c(Prot.combined, Smi.combined), axis = -1) %>% 
    layer_dense(1024, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dense(1, kernel_initializer = "normal", activation ="linear")  
  
  model<-keras_model(inputs =list(InputProt, InputSmi), outputs=Prot.Smi )
  return(model)
  
}



# CNNTXT_v2 --------------------------------------------------------------

CNNTXT_v2<-function(drug.pad, prot.pad){
  InputProt<- layer_input(shape = prot.pad)
  InputSmi<- layer_input(shape = 100)
  
  Prot.Layer1<-InputProt %>% 
    layer_embedding(input_dim =  21, output_dim =  128,mask_zero=TRUE,input_length=prot.pad) %>% 
    layer_conv_1d(filters=32, kernel_size= 4,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=64, kernel_size= 4,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=96, kernel_size= 4,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Prot.Layer2<-InputProt %>% 
    layer_embedding(input_dim =  21, output_dim =  128,mask_zero=TRUE,input_length=prot.pad) %>% 
    layer_conv_1d(filters=32, kernel_size= 8,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=64, kernel_size= 8,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=96, kernel_size= 8,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Prot.Layer3<-InputProt %>% 
    layer_embedding(input_dim =  21, output_dim =  128,mask_zero=TRUE,input_length=prot.pad) %>% 
    layer_conv_1d(filters=32, kernel_size= 12,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=64, kernel_size= 12,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=96, kernel_size= 12,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  #9th June Ingvild changed input_dim to = 31. 
  Smi.Layer1 <-InputSmi %>% 
    layer_embedding(input_dim =  31, output_dim =  128,mask_zero=TRUE,input_length=drug.pad) %>% 
    layer_conv_1d(filters=32, kernel_size= 4,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=64, kernel_size= 4,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=96, kernel_size= 4,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Smi.Layer2 <-InputSmi %>% 
    layer_embedding(input_dim =  31, output_dim =  128,mask_zero=TRUE,input_length=drug.pad) %>% 
    layer_conv_1d(filters=32, kernel_size= 6,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=64, kernel_size= 6,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=96, kernel_size= 6,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Smi.Layer3 <-InputSmi %>% 
    layer_embedding(input_dim =  31, output_dim =  128,mask_zero=TRUE,input_length=drug.pad) %>% 
    layer_conv_1d(filters=32, kernel_size= 8,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=64, kernel_size= 8,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(filters=96, kernel_size= 8,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  
  Prot.Smi<- layer_concatenate(c(Prot.Layer1, Prot.Layer2, Prot.Layer3, Smi.Layer1, Smi.Layer2, Smi.Layer3), axis = -1) %>% 
    layer_dense(1024,  activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dropout(0.1) %>%  
    layer_dense(1024, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dense(1, kernel_initializer = "normal", activation ="linear")  
  
  model<-keras_model(inputs =list(InputProt, InputSmi), outputs=Prot.Smi )
  return(model)
  
}



# DeepDTA-Entropy -------------------------------------------------
mat.vec.mult<- function(args, output_dim){
                entropy.mtx <- tf$tile(tf$expand_dims(args[[2]], -1L), c(1L, 1L,  as.integer(output_dim)))
                out <- tf$math$multiply(args[[1]], entropy.mtx)  
                return(out)
                }


DeepDTA.Entropy<-function(drug.pad, prot.pad, act="linear"){
  
  InputProt <- layer_input(shape = prot.pad)
  P.InputProt <- layer_input(shape = prot.pad)
  InputSmi <- layer_input(shape = drug.pad)
  P.InputSmi <- layer_input(shape = drug.pad)
  
  Prot.Layer<-InputProt %>% 
    layer_embedding(input_dim =  21, output_dim =  128 ,
                    mask_zero=TRUE,
                    input_length=1000, name = "LS")
  
  EntropyCNN.Prot<- list(Prot.Layer, P.InputProt) %>% layer_lambda(mat.vec.mult,arguments =list("output_dim"=128L)) %>% 
    layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%  
    layer_conv_1d(filters=64, kernel_size=8,  activation='relu', padding='same',  strides=1) %>% 
    layer_conv_1d(filters=96, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Smi.Layer<-InputSmi %>% 
    layer_embedding(input_dim =  31, output_dim =  128 ,
                    mask_zero = TRUE,
                    input_length=100)
    EntropyCNN.Smi<- list(Smi.Layer, P.InputSmi) %>% layer_lambda(mat.vec.mult,arguments =list("output_dim"=128L)) %>%   
    layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>% 
    layer_conv_1d(filters=64, kernel_size=6,  activation='relu', padding='same',  strides=1) %>% 
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d()
  
  
  Prot.Smi<-layer_concatenate(c(EntropyCNN.Prot, EntropyCNN.Smi), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") %>%
    layer_dense(1, kernel_initializer = "normal", activation =act)  
  
  model<-keras_model(inputs = c(InputProt, P.InputProt, InputSmi, P.InputSmi), outputs=Prot.Smi )
  return(model)
}

# DeepDTA-Entropy -------------------------------------------------
mat.vec.mult<- function(args, output_dim){
  entropy.mtx <- tf$tile(tf$expand_dims(args[[2]], -1L), c(1L, 1L,  as.integer(output_dim)))
  out <- tf$math$multiply(args[[1]], entropy.mtx)  
  return(out)
}


DeepDTA.Entropy_v2<-function(drug.pad, prot.pad, act="linear"){
  
  InputProt <- layer_input(shape = prot.pad)
  P.InputProt <- layer_input(shape = prot.pad)
  InputSmi <- layer_input(shape = drug.pad)
  P.InputSmi <- layer_input(shape = drug.pad)
  
  Prot.Layer<-InputProt %>% 
    layer_embedding(input_dim =  21, output_dim =  128 ,
                    mask_zero=TRUE,
                    input_length=1000, name = "LS")
  
  EntropyCNN.Prot<- list(Prot.Layer, P.InputProt) %>% layer_lambda(mat.vec.mult,arguments =list("output_dim"=128L)) %>% 
    layer_conv_1d(filters=96, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1) %>%
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Smi.Layer<-InputSmi %>% 
    layer_embedding(input_dim =  31, output_dim =  128 ,
                    mask_zero = TRUE,
                    input_length=100)
  EntropyCNN.Smi<- list(Smi.Layer, P.InputSmi) %>% layer_lambda(mat.vec.mult,arguments =list("output_dim"=128L)) %>%   
    layer_conv_1d(filters=96, kernel_size=4,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=6,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d()
  
  
  Prot.Smi<-layer_concatenate(c(EntropyCNN.Prot, EntropyCNN.Smi), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") %>%
    layer_dense(1, kernel_initializer = "normal", activation =act)  
  
  model<-keras_model(inputs = c(InputProt, P.InputProt, InputSmi, P.InputSmi), outputs=Prot.Smi )
  return(model)
}


# Attention Test1 ---------------------------------------------------------

DTAttention<-function(drug.pad, prot.pad, act="linear",dimension.prot=21, dimension.drug=31, class=FALSE){
  
  InputProt <- layer_input(shape = prot.pad) 
  InputSmi <- layer_input(shape = drug.pad)
  
  Prot.Layer<-InputProt %>% 
    layer_embedding(input_dim =  dimension.prot, output_dim =  96 ,
                    mask_zero=TRUE,
                    input_length=1000, name = "LS") %>% 
    layer_conv_1d(filters=32, kernel_size=8,  activation='relu', padding='same',  strides=1)
  
  Prot.Layer2<- Prot.Layer %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=64, kernel_size=8,  activation='relu', padding='same',  strides=1)
  
  Prot.Layer3<- Prot.Layer2 %>% 
    layer_max_pooling_1d() %>%
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1)
  Prot.Layer4<- Prot.Layer3  %>%  layer_global_max_pooling_1d() 
  
  Smi.Layer<-InputSmi %>% 
    layer_embedding(input_dim =  dimension.drug, output_dim =  96 ,
                    mask_zero = TRUE,
                    input_length=100) %>%  
    layer_conv_1d(filters=32, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer2<- Smi.Layer %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=64, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer3<- Smi.Layer2 %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer4<- Smi.Layer3 %>% layer_global_max_pooling_1d()
  
  Attn1<-list(Prot.Layer, Smi.Layer) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn2<-list(Prot.Layer2, Smi.Layer2) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn3<-list(Prot.Layer3, Smi.Layer3) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  
  
  Prot.Smi<-layer_concatenate(c(Prot.Layer4, Smi.Layer4, Attn1, Attn2, Attn3), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu")
  if(class==FALSE){
    Prot.Smi<- Prot.Smi %>% layer_dense(1, kernel_initializer = "normal", activation =act)  
  }else{
    Prot.Smi<- Prot.Smi %>%layer_dense(2,  activation ="softmax")
  }
  
  model<-keras_model(inputs = c(InputProt,InputSmi), outputs=Prot.Smi )
  return(model)
}


# Attention Test2 ---------------------------------------------------------

DTAttention_v2<-function(drug.pad, prot.pad, act="linear", dimension.prot=21, dimension.drug=31, class=FALSE){
  
  InputProt <- layer_input(shape = prot.pad) 
  InputSmi <- layer_input(shape = drug.pad)
  
  Prot.Layer<-InputProt %>% 
    layer_embedding(input_dim =  dimension.prot, output_dim =  96 ,
                    mask_zero=TRUE,
                    input_length=1000, name = "LS") %>% 
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1)
  
  Prot.Layer2<- Prot.Layer %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1)
  
  Prot.Layer3<- Prot.Layer2 %>% 
    layer_max_pooling_1d() %>%
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1)
  Prot.Layer4<- Prot.Layer3  %>%  layer_global_max_pooling_1d() 
  
  Smi.Layer<-InputSmi %>% 
    layer_embedding(input_dim =  dimension.drug, output_dim =  96 ,
                    mask_zero = TRUE,
                    input_length=100) %>%  
    layer_conv_1d(filters=96, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer2<- Smi.Layer %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer3<- Smi.Layer2 %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer4<- Smi.Layer3 %>% layer_global_max_pooling_1d()
  
  Attn1<-list(Prot.Layer, Smi.Layer) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn2<-list(Prot.Layer2, Smi.Layer2) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn3<-list(Prot.Layer3, Smi.Layer3) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn4<-list(Prot.Layer, Smi.Layer2) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn5<-list(Prot.Layer, Smi.Layer3) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn6<-list(Prot.Layer2, Smi.Layer3) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  
  Prot.Smi<-layer_concatenate(c(Prot.Layer4, Smi.Layer4, Attn1, Attn2, Attn3, Attn4, Attn5, Attn6), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") 
  if(class==FALSE){
    Prot.Smi<- Prot.Smi %>% layer_dense(1, kernel_initializer = "normal", activation =act)  
  }else{
    Prot.Smi<- Prot.Smi %>%layer_dense(2,  activation ="softmax")
  }
  
  
  model<-keras_model(inputs = c(InputProt,InputSmi), outputs=Prot.Smi )
  return(model)
}



# Attention v3 (v1+GEP) ---------------------------------------------------

DTAttention_v3<-function(drug.pad, prot.pad, act="linear",dimension.prot = 21, dimension.drug = 31, class=FALSE){
  
  InputProt <- layer_input(shape = prot.pad) 
  InputSmi <- layer_input(shape = drug.pad)
  
  Prot.Layer<-InputProt %>% 
    layer_embedding(input_dim =  dimension.prot, output_dim =  96 ,
                    #mask_zero=TRUE,
                    input_length=1000, name = "LS") %>% 
    layer_conv_1d(filters=32, kernel_size=8,  activation='relu', padding='same',  strides=1)
  
  Prot.Layer2<- Prot.Layer %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=64, kernel_size=8,  activation='relu', padding='same',  strides=1)
  
  Prot.Layer3<- Prot.Layer2 %>% 
    layer_max_pooling_1d() %>%
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1)
  Prot.Layer4<- Prot.Layer3  %>%  Global_Expectation_Pooling()
  
  Smi.Layer<-InputSmi %>% 
    layer_embedding(input_dim =  dimension.drug, output_dim =  96 ,
                    #mask_zero = TRUE,
                    input_length=100) %>%  
    layer_conv_1d(filters=32, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer2<- Smi.Layer %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=64, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer3<- Smi.Layer2 %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer4<- Smi.Layer3 %>% Global_Expectation_Pooling()
  
  Attn1<-list(Prot.Layer, Smi.Layer) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn2<-list(Prot.Layer2, Smi.Layer2) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn3<-list(Prot.Layer3, Smi.Layer3) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  
  
  Prot.Smi<-layer_concatenate(c(Prot.Layer4, Smi.Layer4, Attn1, Attn2, Attn3), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") 
  if(class==FALSE){
    Prot.Smi<- Prot.Smi %>% layer_dense(1, kernel_initializer = "normal", activation =act)  
  }else{
    Prot.Smi<- Prot.Smi %>%layer_dense(2,  activation ="softmax")
  }
  
  model<-keras_model(inputs = c(InputProt,InputSmi), outputs=Prot.Smi )
  return(model)
}


# Deep DTA GEP ------------------------------------------------------------

DeepDTA.GEP<-function(drug.pad, prot.pad, act="linear", dimension.prot = 21, dimension.drug = 31){
  
  
  InputProt <- layer_input(shape = prot.pad) 
  InputSmi <- layer_input(shape = drug.pad)
  
  Prot.Layer<-InputProt %>% 
    layer_embedding(input_dim =  dimension.prot, output_dim =  128 ,
                    # mask_zero=TRUE,
                    input_length=1000, name = "LS") %>% 
    layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%  
    layer_conv_1d(filters=64, kernel_size=8,  activation='relu', padding='same',  strides=1) %>% 
    layer_conv_1d(filters=96, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
    Global_Expectation_Pooling()
  
  Smi.Layer<-InputSmi %>% 
    layer_embedding(input_dim =  dimension.drug, output_dim =  128 ,
                    #mask_zero = TRUE,
                    input_length=100) %>%  
    layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>% 
    layer_conv_1d(filters=64, kernel_size=6,  activation='relu', padding='same',  strides=1) %>% 
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1) %>%  
    Global_Expectation_Pooling()
  
  
  Prot.Smi<-layer_concatenate(c(Prot.Layer, Smi.Layer), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") %>%
    layer_dense(1, kernel_initializer = "normal", activation =act)  
  
  model<-keras_model(inputs = c(InputProt,InputSmi), outputs=Prot.Smi )
  return(model)
}








# Helper functions block ----------------------------------------------

attention.matrix.gen <- function(arg){
  out<-tf$norm(arg[[1]]-arg[[2]], axis=c(3))
  out<-1/(1-out)
  return(out)
}

# Zero Tensor 
zerotensor<-function(arg){
  zeros<-k_zeros_like(arg)
  
  return(zeros)
}

# Trainable layer

CustomLayer <- R6::R6Class("CustomLayer",
                           
                           inherit = KerasLayer,
                           
                           public = list(
                             
                             output_dim = NULL,
                             
                             kernel = NULL,
                             
                             initialize = function(output_dim) {
                               self$output_dim <- output_dim
                             },
                             
                             build = function(input_shape) {
                               self$kernel <- self$add_weight(
                                 name = 'kernel', 
                                 shape = list(input_shape[[3]], self$output_dim),
                                 initializer = initializer_random_normal(),
                                 trainable = TRUE
                               )
                             },
                             
                             call = function(x, mask = NULL) {
                               k_dot(x, self$kernel)
                             },
                             
                             compute_output_shape = function(input_shape) {
                               list(input_shape[[3]], self$output_dim)
                             }
                           )
)

layer_dotproduct <- function(object, output_dim, name = NULL, trainable = TRUE) {
  create_layer(CustomLayer, object, list(
    output_dim = as.integer(output_dim),
    name = name,
    trainable = trainable
  ))
}




# Distance matrix generator 

distance.matrix.gen <- function(arg){
  # input1 <- tf$cast(tf$transpose(arg[[1]], perm=list(0L,2L,1L)), tf$float64)
  # input2 <- tf$cast(tf$transpose(arg[[2]], perm=list(0L,2L,1L)), tf$float64)
  
  input1 <- tf$cast(arg[[1]], tf$float64)
  input2 <- tf$cast(arg[[2]], tf$float64)
  
  shapeInput1 <- tf$shape(input1, out_type=tf$int32)
  shapeInput2 <- tf$shape(input2, out_type=tf$int32)
  
  na <- tf$expand_dims(tf$reduce_sum(tf$square(input1), 2L), 2L)
  nb <- tf$expand_dims(tf$reduce_sum(tf$square(input2), 2L), 2L)
  
  na_ones<- tf$ones_like(tf$transpose(nb, perm = list(0L,2L,1L)))
  nb_ones<- tf$ones_like(tf$transpose(na, perm = list(0L,2L,1L)))
  
  p1 <- tf$matmul(na, na_ones)
  p2 <- tf$matmul(nb, nb_ones)
  
  p2 <- tf$transpose(p2,perm=list(0L,2L,1L))
  
  out <- tf$sqrt(tf$add(p1, p2) - tf$matmul(input1, input2, transpose_b=TRUE) + tf$matmul(input1, input2, transpose_b=TRUE))
  #Inverse distance->attention 1/(1+distance)
  out.ones<-tf$ones_like(out)
  out<-tf$add(out, out.ones)
  #out<-tf$div(out.ones, out) #TF<2
  out<-tf$compat$v1$div(out.ones, out) #TF>2
  return(out) 
}


# Matrix conformation ABCNN1 

matrix_merger<-function(arg){
  input1 <- tf$cast(arg[[1]], tf$float64)
  input2 <- tf$cast(arg[[2]], tf$float64)
  input1<-k_expand_dims(input1)
  input2<-k_expand_dims(input2)
  out<-layer_concatenate(list(input1,input2))
  return(out)
}


# Squezeer dimensions 

squezeer<- function(arg){
  
  out<-tf$squeeze(arg, axis=-2)
  return(out)
}

# ABCNNv1 -----------------------------------------------------------------


ABCNN1<-function(prot.pad=1000,
                 drug.pad=100,
                 embd.prot=128,
                 embd.drug=128,
                 aa=21,
                 smi.com=32,
                 filters1=32,
                 prot.win=4,
                 drug.win=6,
                 act="linear",
                 class=FALSE){
  InputProt<- layer_input(shape = prot.pad)
  InputSmi<- layer_input(shape = drug.pad)
  
  Prot.Layer1<-InputProt %>% 
    layer_embedding(input_dim =  aa, output_dim =  embd.prot,
                    #mask_zero=TRUE,
                    input_length=prot.pad) %>% 
    layer_conv_1d(filters=filters1, kernel_size= prot.win,  activation='relu', padding='same',  strides=1) %>% 
    layer_conv_1d(filters=filters1*2, kernel_size= prot.win,  activation='relu', padding='same',  strides=1) %>% 
    layer_conv_1d(filters=filters1*3, kernel_size= prot.win,  activation='relu', padding='same',  strides=1)
  
  Drug.Layer1<-InputSmi %>% 
    layer_embedding(input_dim =  smi.com, output_dim =  embd.drug,
                    #mask_zero=TRUE,
                    input_length=drug.pad) %>% 
    layer_conv_1d(filters=filters1, kernel_size= drug.win,  activation='relu', padding='same',  strides=1) %>% 
    layer_conv_1d(filters=filters1*2, kernel_size= drug.win,  activation='relu', padding='same',  strides=1) %>% 
    layer_conv_1d(filters=filters1*3, kernel_size= drug.win,  activation='relu', padding='same',  strides=1)
  
  
  Attention<- list(Prot.Layer1, Drug.Layer1) %>% layer_lambda(distance.matrix.gen)
  
  W1<-Attention %>% layer_dotproduct(output_dim = filters1*3)
  #Transpose before merging
  W2<-layer_permute(Attention, dims = c(2,1)) %>% layer_dotproduct(output_dim = filters1*3)
  
  
  W1_CNN<-list(Prot.Layer1, W1) %>% 
    layer_lambda(matrix_merger) %>% 
    layer_conv_2d(1,kernel_size = 1) %>% 
    layer_reshape(target_shape = c(prot.pad, filters1*3)) %>% 
    layer_global_max_pooling_1d()
  
  W2_CNN<-list(Drug.Layer1, W2) %>% 
    layer_lambda(matrix_merger) %>% 
    layer_conv_2d(1,kernel_size = 1) %>% 
    layer_reshape(target_shape = c(drug.pad, filters1*3)) %>% 
    layer_global_max_pooling_1d()
  
  Prot.Smi<-layer_concatenate(c(W1_CNN, W2_CNN), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") 
  if(class==FALSE){
    Prot.Smi<- Prot.Smi %>% layer_dense(1, kernel_initializer = "normal", activation =act)  
  }else{
    Prot.Smi<- Prot.Smi %>%layer_dense(2,  activation ="softmax")
  }
  
  model<-keras_model(inputs = c(InputProt,InputSmi), outputs=Prot.Smi )
  return(model)
  
}


# ABCNNv1_multiAttention -----------------------------------------------------------------

ABCNN1_MA<-function(prot.pad=1000,
                    drug.pad=100,
                    embd.prot=128,
                    embd.drug=128,
                    aa=21,
                    smi.com=32,
                    filters1=32,
                    prot.win=4,
                    drug.win=6,
                    act="linear"){
  InputProt<- layer_input(shape = prot.pad)
  InputSmi<- layer_input(shape = drug.pad)
  
  Prot.Layer1<-InputProt %>% 
    layer_embedding(input_dim =  aa, output_dim =  embd.prot,
                    #mask_zero=TRUE,
                    input_length=prot.pad) %>% 
    layer_conv_1d(filters=filters1, kernel_size= prot.win,  activation='relu', padding='same',  strides=1)
  
  Drug.Layer1<-InputSmi %>% 
    layer_embedding(input_dim =  smi.com, output_dim =  embd.drug,
                    #mask_zero=TRUE,
                    input_length=drug.pad) %>% 
    layer_conv_1d(filters=filters1, kernel_size= drug.win,  activation='relu', padding='same',  strides=1)
  
  
  Attention<- list(Prot.Layer1, Drug.Layer1) %>% layer_lambda(distance.matrix.gen, name = "Att_1")
  
  W1<-Attention %>% layer_dotproduct(output_dim = filters1)
  W2<-layer_permute(Attention, dims = c(2,1)) %>% layer_dotproduct(output_dim = filters1)
  
  Prot.Layer2<-list(Prot.Layer1, W1) %>% 
    layer_lambda(matrix_merger) %>% 
    layer_conv_2d(1,kernel_size = 1) %>% 
    layer_reshape(target_shape = c(prot.pad, filters1)) %>% 
    layer_conv_1d(filters=filters1*2, kernel_size= prot.win,  activation='relu', padding='same',  strides=1)
  
  Drug.Layer2<-list(Drug.Layer1, W2) %>% 
    layer_lambda(matrix_merger) %>% 
    layer_conv_2d(1,kernel_size = 1) %>% 
    layer_reshape(target_shape = c(drug.pad, filters1)) %>% 
    layer_conv_1d(filters=filters1*2, kernel_size= prot.win,  activation='relu', padding='same',  strides=1)
  
  Attention2<-list(Prot.Layer2, Drug.Layer2) %>% layer_lambda(distance.matrix.gen,name = "Att_2")
  
  W1_2<-Attention2 %>% layer_dotproduct(output_dim = filters1*2)
  W2_2<-layer_permute(Attention2, dims = c(2,1)) %>% layer_dotproduct(output_dim = filters1*2)
  
  Prot.Layer3<-list(Prot.Layer2, W1_2) %>% 
    layer_lambda(matrix_merger) %>% 
    layer_conv_2d(1,kernel_size = 1) %>% 
    layer_reshape(target_shape = c(prot.pad, filters1*2)) %>% 
    layer_conv_1d(filters=filters1*3, kernel_size= prot.win,  activation='relu', padding='same',  strides=1)
  
  Drug.Layer3<-list(Drug.Layer2, W2_2) %>% 
    layer_lambda(matrix_merger) %>% 
    layer_conv_2d(1,kernel_size = 1) %>% 
    layer_reshape(target_shape = c(drug.pad, filters1*2)) %>% 
    layer_conv_1d(filters=filters1*3, kernel_size= prot.win,  activation='relu', padding='same',  strides=1)
  
  
  Attention3<-list(Prot.Layer3, Drug.Layer3) %>% layer_lambda(distance.matrix.gen, name = "Att_3")
  
  W1_3<-Attention2 %>% layer_dotproduct(output_dim = filters1*3)
  W2_3<-layer_permute(Attention2, dims = c(2,1)) %>% layer_dotproduct(output_dim = filters1*3)
  
  Prot.Layer4<-list(Prot.Layer3, W1_3) %>% 
    layer_lambda(matrix_merger) %>% 
    layer_conv_2d(1,kernel_size = 1) %>% 
    layer_reshape(target_shape = c(prot.pad, filters1*3)) %>% 
    layer_global_max_pooling_1d()
  
  Drug.Layer4<-list(Drug.Layer3, W2_3) %>% 
    layer_lambda(matrix_merger) %>% 
    layer_conv_2d(1,kernel_size = 1) %>% 
    layer_reshape(target_shape = c(drug.pad, filters1*3)) %>% 
    layer_global_max_pooling_1d()
  
  Prot.Smi<-layer_concatenate(c(Prot.Layer4, Drug.Layer4), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") %>%
    layer_dense(1, kernel_initializer = "normal", activation =act)  
  
  model<-keras_model(inputs = c(InputProt,InputSmi), outputs=Prot.Smi )
  return(model)
  
}


# Attention Test2B ---------------------------------------------------------

DTAttention_v2B<-function(drug.pad, prot.pad, act="linear",dimension.prot = 21, dimension.drug = 31){
  
  InputProt <- layer_input(shape = prot.pad) 
  InputSmi <- layer_input(shape = drug.pad)
  
  Prot.Layer<-InputProt %>% 
    layer_embedding(input_dim =  dimension.prot, output_dim =  96 ,
                    #mask_zero=TRUE,
                    input_length=1000, name = "LS") %>% 
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1)
  
  Prot.Layer2<- Prot.Layer %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1)
  
  Prot.Layer3<- Prot.Layer2 %>% 
    layer_max_pooling_1d() %>%
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1)
  Prot.Layer4<- Prot.Layer3  %>%  layer_global_max_pooling_1d() 
  
  Smi.Layer<-InputSmi %>% 
    layer_embedding(input_dim =  dimension.drug, output_dim =  96 ,
                    #mask_zero = TRUE,
                    input_length=100) %>%  
    layer_conv_1d(filters=96, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer2<- Smi.Layer %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer3<- Smi.Layer2 %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer4<- Smi.Layer3 %>% layer_global_max_pooling_1d()
  
  Attn1<-list(Prot.Layer, Smi.Layer) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn2<-list(Prot.Layer2, Smi.Layer2) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn3<-list(Prot.Layer3, Smi.Layer3) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn4<-list(Prot.Layer, Smi.Layer2) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn5<-list(Prot.Layer, Smi.Layer3) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn6<-list(Prot.Layer2, Smi.Layer3) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  
  Attn1B<-list(Smi.Layer,Prot.Layer) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn2B<-list(Smi.Layer2,Prot.Layer2) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn3B<-list(Smi.Layer3,Prot.Layer3) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn4B<-list(Smi.Layer2,Prot.Layer) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn5B<-list(Smi.Layer3,Prot.Layer) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn6B<-list(Smi.Layer3,Prot.Layer2) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  
  Prot.Smi<-layer_concatenate(c(Prot.Layer4, Smi.Layer4, Attn1, Attn2, Attn3, Attn4, Attn5, Attn6,
                                Attn1B, Attn2B, Attn3B, Attn4B, Attn5B, Attn6B), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") %>%
    layer_dense(1, kernel_initializer = "normal", activation =act)  
  
  model<-keras_model(inputs = c(InputProt,InputSmi), outputs=Prot.Smi )
  return(model)
}


# Attention RNN ---------------------------------------------------------

DTAttention_RNN<-function(drug.pad, prot.pad, act="linear", gru.units=128){
  
  InputProt <- layer_input(shape = prot.pad) 
  InputSmi <- layer_input(shape = drug.pad)
  
  Prot.Layer<-InputProt %>% 
    layer_embedding(input_dim =  21, output_dim =  96 ,
                    #mask_zero=TRUE,
                    input_length=1000, name = "LS.Pro") %>% 
    bidirectional(layer_cudnn_gru(units = gru.units, return_sequences = TRUE, return_state = TRUE))
  
  Smi.Layer<-InputSmi %>% 
    layer_embedding(input_dim =  31, output_dim =  96 ,
                    #mask_zero=TRUE,
                    input_length=100, name = "LS.Smi") %>% 
    bidirectional(layer_cudnn_gru(units = gru.units, return_sequences = TRUE, return_state = TRUE))
  
  Att<- layer_attention(list(Prot.Layer[[1]], Smi.Layer[[1]])) %>% layer_global_max_pooling_1d() %>% layer_dense(64)
  Att2<- layer_attention(list(Smi.Layer[[1]], Prot.Layer[[1]])) %>% layer_global_max_pooling_1d()%>% layer_dense(64)
  
  Smi2<-Smi.Layer[[2]] %>% layer_dense(64)
  Prot2<-Prot.Layer[[2]]%>% layer_dense(64)
  
  
  Prot.Smi<-layer_concatenate(c(Smi2, Prot2, Att, Att2), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") %>%
    layer_dense(1, kernel_initializer = "normal", activation =act)  
  
  model<-keras_model(inputs = c(InputProt,InputSmi), outputs=Prot.Smi )
  return(model)
}


# CNNTXT M66 --------------------------------------------------------------


CNNTXT_66<- function(input_dim.1=21,
                     input_dim.2=21,
                     input_dim.3=21,
                     embd.prot=64,
                     pad.prot.1=1000,
                     pad.prot.2=1000,
                     pad.prot.3=1000,
                     chan.prot=19,
                     inc.chan.prot=2,
                     win.prot=6,
                     poolsize.prot=2,
                     embd.smi=64,
                     chan.smi=21,
                     inc.chan.smi=3,
                     win.smi=8,
                     poolsize.smi=4,
                     nn.layer1=600,
                     nn.layer2=203,
                     nn.layer3=50,
                     input_dim.drug=31,
                     
                     act="linear"){
  
  InputProt.1 <- layer_input(shape = c(pad.prot.1))
  InputProt.2 <- layer_input(shape = c(pad.prot.2))
  InputProt.3 <- layer_input(shape = c(pad.prot.3))
  
  InputSmi <- layer_input(shape = c(100))
  
  
  #Ingvild June 15th added 2 more protein and smiles layers.
  #Ingvild 24th added max_pooling layers
  Prot.Layer1<-InputProt.1 %>% 
    layer_embedding(input_dim =  input_dim.1, output_dim =  embd.prot,mask_zero=TRUE,input_length=pad.prot.1) %>% 
    layer_conv_1d(filters=chan.prot, kernel_size= win.prot,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(poolsize.prot) %>%
    layer_conv_1d(filters=(chan.prot*inc.chan.prot), kernel_size= win.prot,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(poolsize.prot) %>%
    layer_conv_1d(filters=(chan.prot*(inc.chan.prot*2)), kernel_size= win.prot,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Prot.Layer2<-InputProt.2 %>% 
    layer_embedding(input_dim =  input_dim.2, output_dim =  embd.prot,mask_zero=TRUE,input_length=pad.prot.2) %>% 
    layer_conv_1d(filters=chan.prot, kernel_size= win.prot,  activation='relu', padding='same',  strides=1) %>%  
    layer_max_pooling_1d(poolsize.prot) %>%
    layer_conv_1d(filters=(chan.prot*inc.chan.prot), kernel_size=win.prot,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(poolsize.prot) %>%
    layer_conv_1d(filters=(chan.prot*(inc.chan.prot*2)), kernel_size=win.prot,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Prot.Layer3<-InputProt.3 %>% 
    layer_embedding(input_dim =  input_dim.3, output_dim =  embd.prot,mask_zero=TRUE,input_length=pad.prot.3) %>% 
    layer_conv_1d(filters=chan.prot, kernel_size= win.prot,  activation='relu', padding='same',  strides=1) %>%  
    layer_max_pooling_1d(poolsize.prot) %>%
    layer_conv_1d(filters=(chan.prot*inc.chan.prot), kernel_size=win.prot,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(poolsize.prot) %>%
    layer_conv_1d(filters=(chan.prot*(inc.chan.prot*2)), kernel_size=win.prot,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  #9th June Ingvild changed input_dim to = 31. 
  Smi.Layer <-InputSmi %>% 
    layer_embedding(input_dim = input_dim.drug, output_dim =  embd.smi,mask_zero = TRUE,input_length=100) %>%  
    layer_conv_1d(filters=chan.smi, kernel_size= win.smi,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(poolsize.smi) %>%
    layer_conv_1d(filters=(chan.smi*inc.chan.smi), kernel_size= win.smi,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d(poolsize.smi) %>%
    layer_conv_1d(filters=(chan.smi*(inc.chan.smi*2)), kernel_size= win.smi,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d()
  
  Prot.Smi<-layer_concatenate(c(Prot.Layer1, Prot.Layer2, Prot.Layer3, Smi.Layer), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(nn.layer1,  activation = "sigmoid") %>%
    layer_dropout(0.1) %>%
    layer_dense(nn.layer2, activation = "sigmoid") %>%
    layer_dropout(0.1) %>%
    layer_dense(nn.layer3, activation = "sigmoid") %>%
    layer_dense(1, kernel_initializer = "normal", activation =act)  
  
  
  model<-keras_model(inputs = c(InputProt.1,InputProt.2, InputProt.3,InputSmi), outputs=Prot.Smi )
  return(model)
}



# DeepDTA_MP model generator -------------------------------------------------

DeepDTAMP<-function(drug.pad, prot.pad, act="linear", dimension.prot=21, dimension.drug=31){
  
  InputProt <- layer_input(shape = prot.pad) 
  InputSmi <- layer_input(shape = drug.pad)
  
  Prot.Layer<-InputProt %>% 
    layer_embedding(input_dim =  dimension.prot, output_dim =  128 ,
                    mask_zero=TRUE,
                    input_length=1000, name = "LS") %>% 
    layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%  
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=64, kernel_size=8,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Smi.Layer<-InputSmi %>% 
    layer_embedding(input_dim =  dimension.drug, output_dim =  128 ,
                    mask_zero = TRUE,
                    input_length=100) %>%  
    layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=64, kernel_size=6,  activation='relu', padding='same',  strides=1) %>%
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d()
  
  
  Prot.Smi<-layer_concatenate(c(Prot.Layer, Smi.Layer), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") %>%
    layer_dense(1, kernel_initializer = "normal", activation =act)  
  
  model<-keras_model(inputs = c(InputProt,InputSmi), outputs=Prot.Smi )
  return(model)
}



# DeepDTA_AP model generator -------------------------------------------------

DeepDTA_AP<-function(drug.pad, prot.pad, act="linear", dimension.prot=21, dimension.drug=31){
  
  InputProt <- layer_input(shape = prot.pad) 
  InputSmi <- layer_input(shape = drug.pad)
  
  Prot.Layer<-InputProt %>% 
    layer_embedding(input_dim =  dimension.prot, output_dim =  128 ,
                    mask_zero=TRUE,
                    input_length=1000, name = "LS") %>% 
    layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%  
    layer_average_pooling_1d() %>% 
    layer_batch_normalization() %>% 
    layer_conv_1d(filters=64, kernel_size=8,  activation='relu', padding='same',  strides=1) %>% 
    layer_average_pooling_1d() %>% 
    layer_batch_normalization() %>% 
    layer_conv_1d(filters=96, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Smi.Layer<-InputSmi %>% 
    layer_embedding(input_dim =  dimension.drug, output_dim =  128 ,
                    mask_zero = TRUE,
                    input_length=100) %>%  
    layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>% 
    layer_average_pooling_1d() %>%
    layer_batch_normalization() %>% 
    layer_conv_1d(filters=64, kernel_size=6,  activation='relu', padding='same',  strides=1) %>%
    layer_average_pooling_1d() %>%
    layer_batch_normalization() %>% 
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d()
  
  
  Prot.Smi<-layer_concatenate(c(Prot.Layer, Smi.Layer), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_batch_normalization() %>% 
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_batch_normalization() %>% 
    layer_dense(512, activation = "relu") %>%
    layer_dense(1, kernel_initializer = "normal", activation =act)  
  
  model<-keras_model(inputs = c(InputProt,InputSmi), outputs=Prot.Smi )
  return(model)
}

# Trainer function --------------------------------------------------------

model_trainer<- function(normalize=TRUE,
                         CV=TRUE,
                         activity.data=df.activity,
                         drug.arr=drug.arr,
                         prot.arr=prot.arr.1,
                         model=model,
                         Batch=300,
                         Epochs=100,
                         patience=10,
                         path=path,
                         use.pushover=TRUE,
                         model.name=model.name){
  
  df.activity<-activity.data
  prot.arr.1<.prot.arr
  
  if(normalize){
    #Normalization
    df.activity$Activity2<-df.activity$Activity
    df.activity$Activity2 <- (df.activity$Activity2 - min(df.activity$Activity2))/(max(df.activity$Activity2)-min(df.activity$Activity2))
    df.activity$Activity<-df.activity$Activity2
    df.activity$Activity2<-NULL
  }
  
  val.loss<-vector()
  train.loss<-vector()
  val.ci <-vector()
  train.ci <- vector()
  
  CV<-1
  if(CV) CV<-nrow(samples)
  
  for (j in 1:CV) {
    df.training<- df.activity[-as.numeric(unlist(samples[j,])),] #Nacho 8th June 2020 changed to samples[j,]
    df.validation<- df.activity[as.numeric(unlist(samples[j,])),] #Nacho 8th June 2020 changed to samples[j,]
    
    #Nacho 9th June 2020
    #Problem with the replacement of arrays, if we replace them we can't get them back easily
    # change the name to drug_train and prot_train
    drug_test <- drug.arr[as.numeric(unlist(samples[j,])),] #Nacho 8th June 2020 changed to samples[j,]
    drug_train <- drug.arr[-as.numeric(unlist(samples[j,])),] #Nacho 8th June 2020 changed to samples[j,]
    
    prot_test.1 <- prot.arr.1[as.numeric(unlist(samples[j,])),] #Nacho 8th June 2020 changed to samples[j,]
    prot_train.1 <- prot.arr.1[-as.numeric(unlist(samples[j,])),]
    
    prot_test.1<- matrix(as.numeric(prot_test.1), ncol = pad.prot1)
    prot_train.1<-matrix(as.numeric(prot_train.1), ncol = pad.prot1)
    
    gc()
    
    
    
    #Model compile
    opt <- optimizer_adam( lr= 0.001 , decay = 1e-07 )
    
    #opt.adagrad <- optimizer_adagrad(lr = 0.01, epsilon = NULL, decay = 0,
    #                                clipnorm = NULL, clipvalue = NULL)
    
    #opt.nadam <- optimizer_nadam(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999,
    #                             epsilon = NULL, schedule_decay = 0.004, clipnorm = NULL,
    #                            clipvalue = NULL)
    
    compile(model,optimizer = opt, loss = 'mean_squared_error', metrics = cindex_score)
    
    #Model fit
    #Ingvild: Altered the code as to get new logs each time. 
    
    #date<-as.character(date())
    #logs<-gsub(" ","_",date)
    #logs<-gsub(":",".",logs)
    #logs<-paste("logs/",logs,sep = "")
    
    callbacks.1 <- list(callback_early_stopping(monitor="val_python_function", patience=patience, verbose=0, mode='max', restore_best_weights = TRUE))
    try(rm(history))
    history<-model %>% fit(x= list(prot_train.1, drug_train),
                           y=df.training$Activity,
                           batch_size=300, #06082020 Nacho: New BS
                           epoch=200,
                           validation_data = list(list(prot_test.1, drug_test),df.validation$Activity),
                           callbacks =callbacks.1, 
                           view_metrics=FALSE,                                                                                        
                           shuffle=TRUE)  
    
    list.save(history,paste(path,date,"-history_",model.name,"_fold_",j,".yaml", sep = "" ), type = "yaml")
    if(j==1) save_model_weights_hdf5(model, paste(path,date,"Weights",model.name,"_fold1",".h5", sep = "" ))
    
    
    scores <- history[["metrics"]]
    train.loss[j]<-min(scores$loss)
    val.loss[j]<-min(scores$val_loss)
    train.ci[j]<-max(scores$python_function)
    val.ci[j]<-max(scores$val_python_function)
    gc()
    if(use.pushover) try(pushover(paste(model.name, "fold", j,  "CI",val.ci[j])))
  }
  
  if(use.pushover) try(pushover(paste(model.name,"is completed! Mean",  "equals",mean(val.ci, na.rm = TRUE))))
  
  resultsDeepDTA<-c(mean(train.loss, na.rm = TRUE), sd(train.loss, na.rm = TRUE),
                    mean(train.ci, na.rm = TRUE), sd(train.ci, na.rm = TRUE),
                    mean(val.loss, na.rm = TRUE), sd(val.loss, na.rm = TRUE),
                    mean(val.ci, na.rm = TRUE), sd(val.ci, na.rm = TRUE))
  
  
  resultsDeepDTA<-as.data.frame(t(resultsDeepDTA))
  colnames(resultsDeepDTA)<-c("train.loss", "sd.train.loss", "train.ci", "sd.train.ci", "val.loss", "sd.val.loss", "val.ci", "sd.val.ci")
  write.csv(resultsDeepDTA, paste(path,date,model.name,"Results.csv", sep = ""))
  
  
  
}




# BlosumGen ---------------------------------------------------------------

blosum_gen<-function(BlosumInput, base=4){
  
  BLOSUMX <-  normalize.row(Data = BlosumInput, base = base)
  
  rownames(BLOSUMX) = rownames(BlosumInput)
  colnames(BLOSUMX) = colnames(BlosumInput)
  rownames.bl3 <- vector()
 
  for(i in 1:nrow(BLOSUMX)){
    aa.number <- aa.number[1:20,]
    aa <- rownames(BLOSUMX)
    aa <- aa[i]
    which.aa <- which(aa.number$ID == aa)
    num <- aa.number$Num[which.aa]
    rownames.bl3[i] <- num
  }
  
  rownames(BLOSUMX) = rownames.bl3
  colnames(BLOSUMX) = rownames.bl3
  
  return(list(BLOSUMX, rownames.bl3))
}




# DeepDTA FingerPrints ----------------------------------------------------


DeepDTA_FP<-function(drug.pad, prot.pad, act="linear", dimension.prot=21, dimension.drug=128*3){
  
  InputProt <- layer_input(shape = prot.pad) 
  InputSmi <- layer_input(shape = 512)
  
  Prot.Layer<-InputProt %>% 
    layer_embedding(input_dim =  dimension.prot, output_dim =  128 ,
                    #mask_zero=TRUE,
                    input_length=1000, name = "LS") %>% 
    layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%  
    layer_conv_1d(filters=64, kernel_size=8,  activation='relu', padding='same',  strides=1) %>% 
    layer_conv_1d(filters=96, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
    layer_global_max_pooling_1d() 
  
  Smi.Layer<-InputSmi %>% layer_dense(512) %>%
    layer_dense(128) %>% 
    layer_dense(128)
  
  Prot.Smi<-layer_concatenate(c(Prot.Layer, Smi.Layer), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") %>%
    layer_dense(1, kernel_initializer = "normal", activation =act)  
  
  model<-keras_model(inputs = c(InputProt,InputSmi), outputs=Prot.Smi )
  return(model)
}
# Pure Attention ---------------------------------------------------------

DTAttention_pure<-function(drug.pad, prot.pad, act="linear", dimension.prot=21, dimension.drug=31){
  
  InputProt <- layer_input(shape = prot.pad) 
  InputSmi <- layer_input(shape = drug.pad)
  
  Prot.Layer<-InputProt %>% 
    layer_embedding(input_dim =  dimension.prot, output_dim =  96 ,
                    mask_zero=TRUE,
                    input_length=1000, name = "LS") %>% 
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1)
  
  Prot.Layer2<- Prot.Layer %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1)
  
  Prot.Layer3<- Prot.Layer2 %>% 
    layer_max_pooling_1d() %>%
    layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1)
  Prot.Layer4<- Prot.Layer3  %>%  layer_global_max_pooling_1d() 
  
  Smi.Layer<-InputSmi %>% 
    layer_embedding(input_dim =  dimension.drug, output_dim =  96 ,
                    mask_zero = TRUE,
                    input_length=100) %>%  
    layer_conv_1d(filters=96, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer2<- Smi.Layer %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer3<- Smi.Layer2 %>% 
    layer_max_pooling_1d() %>% 
    layer_conv_1d(filters=96, kernel_size=6,  activation='relu', padding='same',  strides=1)
  
  Smi.Layer4<- Smi.Layer3 %>% layer_global_max_pooling_1d()
  
  Attn1<-list(Prot.Layer, Smi.Layer) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn2<-list(Prot.Layer2, Smi.Layer2) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn3<-list(Prot.Layer3, Smi.Layer3) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn4<-list(Prot.Layer, Smi.Layer2) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn5<-list(Prot.Layer, Smi.Layer3) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  Attn6<-list(Prot.Layer2, Smi.Layer3) %>% layer_attention(trainable = TRUE) %>% layer_global_max_pooling_1d()
  
  Prot.Smi<-layer_concatenate(c(Attn1, Attn2, Attn3, Attn4, Attn5, Attn6), axis = -1) %>% 
    #New hyperparams defining the number of neurons
    layer_dense(1024,  activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(1024, activation = "relu") %>%
    layer_dropout(0.1) %>%
    layer_dense(512, activation = "relu") %>%
    layer_dense(1, kernel_initializer = "normal", activation =act)  
  
  model<-keras_model(inputs = c(InputProt,InputSmi), outputs=Prot.Smi )
  return(model)
}


