library("seqinr")
library("keras")
library("tensorflow")


input.len<-51
mask.pos<-25

human<-read.fasta("/home/nacho/DNA-RNN/CDS_Human.fna")
bat<-read.fasta("/home/nacho/DNA-RNN/CDS_Rhinolophus sinicus.fna")

genome<-human
train.size<-61111

genome.x<-list()
genome.y<-list()

for (i in 1:length(genome)) {
  continue = TRUE  
  while (continue) {
    genome.x[[length(genome.x)+1]] <- genome[[i]][1:input.len]
    genome.y[[length(genome.x)+1]] <- genome[[i]][mask.pos]
    genome[[i]]<-genome[[i]][-1]
    if(length(genome[[i]])<input.len | length(genome.x) == train.size) continue<-FALSE
  }
  if(length(genome.x) == train.size) break()
}


genome.x<-as.data.frame(do.call( rbind, genome.x))
genome.y<-unlist(genome.y)
genome.x[,mask.pos]<-"x"

genome.x<-matrix(match(as.matrix(genome.x), c("x","a","t","c","g"))-1, ncol = ncol(genome.x)) 
genome.y<-match(as.matrix(genome.y), c("a","t","c","g"))-1
genome.y<-to_categorical(genome.y)
genome.y<-genome.y[,-1]

InputDNA<-layer_input(shape = (ncol(genome.x))) 

Layers<-InputDNA %>% 
  layer_embedding(input_dim =  5, output_dim =  42 ,
                  mask_zero=TRUE,
                  input_length=input.len, name = "Embd") %>% 
  layer_conv_1d(filters=32, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%  
  layer_conv_1d(filters=64, kernel_size=8,  activation='relu', padding='same',  strides=1) %>% 
  layer_conv_1d(filters=96, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
  layer_global_max_pooling_1d() %>% 
  layer_dense(units = 128) %>% 
  layer_dense(units = 28) %>% 
  layer_dense(units = 4, activation = "softmax")

model<-keras_model(inputs = InputDNA, outputs=Layers )
summary(model)



model %>% compile(
  optimizer = 'adagrad',
  loss = "categorical_crossentropy")

history<-model %>% fit(
  as.matrix(genome.x),
  genome.y,
  epochs = 50,
  batch_size = 100,
  callbacks=list(callback_early_stopping(monitor='val_loss',
                                         patience=3,
                                         verbose=0,
                                         mode='auto',
                                         restore_best_weights = TRUE))
)
  
