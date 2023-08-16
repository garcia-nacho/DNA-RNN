library("seqinr")
library("keras")
library("tensorflow")
library("caret")

input.len<-51
mask.pos<-25

human<-read.fasta("/home/nacho/DNA-RNN-main//CDS_Human.fna")
bat<-read.fasta("/home/nacho/DNA-RNN-main/CDS_Rhinolophus sinicus.fna")

human.val<- human[c(2001:3000)]
bat.val<-human[c(2001:3000)]

human<-human[-c(2001:3000)]
bat<-bat[-c(2001:3000)]

for (sp in c("human","bat")) {
  if(sp=="human") genome<-human
  if(sp=="human") genome<-bat

  train.size<-800000

genome.x<-list()
genome.y<-list()

for (i in 1:length(genome)) {
  continue = TRUE  
  while (continue) {
    genome.x[[length(genome.x)+1]] <- genome[[i]][1:input.len]
    genome.y[[length(genome.y)+1]] <- genome[[i]][mask.pos]
    genome[[i]]<-genome[[i]][-1]
    if(length(genome[[i]])<input.len | length(genome.x) == train.size) continue<-FALSE
  }
  if(length(genome.x) == train.size) break()
}

print(paste("Completed ",i," elements",sep = ""))

genome.x<-as.data.frame(do.call( rbind, genome.x))
genome.y<-unlist(genome.y)
genome.x[,mask.pos]<-"x"


genome.x<-matrix(match(as.matrix(genome.x), c("x","a","t","c","g"))-1, ncol = ncol(genome.x)) 
genome.y_base<-genome.y
genome.y<-match(genome.y, c("a","t","c","g"))-1

genome.y<-to_categorical(genome.y)

training<-sample(c(1:nrow(genome.x)),round(nrow(genome.x)/20))

genome.x.test<-genome.x[training,]
genome.y.test<-genome.y[training,]
genom.y_base.test<-genome.y_base[training]

genome.x<-genome.x[-training,]
genome.y<-genome.y[-training,]
genome.y_base<-genome.y_base[-training]


InputDNA<-layer_input(shape = (ncol(genome.x))) 

Layers<-InputDNA %>% 
  layer_embedding(input_dim =  5, output_dim =  64 ,
                  mask_zero=FALSE,
                  input_length=input.len, name = "Embd") %>% 
  layer_conv_1d(filters=64, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%  
  layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1) %>% 
  layer_conv_1d(filters=128, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
  layer_conv_1d(filters=64, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%  
  layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1) %>% 
  layer_conv_1d(filters=128, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
  layer_conv_1d(filters=64, kernel_size=4,  activation='relu', padding='same',  strides=1) %>%  
  layer_conv_1d(filters=96, kernel_size=8,  activation='relu', padding='same',  strides=1) %>% 
  layer_conv_1d(filters=128, kernel_size=12,  activation='relu', padding='same',  strides=1) %>%  
  layer_global_max_pooling_1d() %>% 
  layer_dense(units = 64) %>% 
  layer_dense(units = 28) %>% 
  layer_dense(units = 4, activation = "softmax")


model<-keras_model(inputs = InputDNA, outputs=Layers )
summary(model)





model %>% compile(
  optimizer = 'adagrad',
  loss = "categorical_crossentropy",
  metrics = "categorical_accuracy")

history<-model %>% fit(
  as.matrix(genome.x),
  matrix(genome.y, ncol = 4),
  epochs = 50,
  batch_size = 100,
  validation_data = list(as.matrix(genome.x.test), matrix(genome.y.test,ncol = 4)),
  callbacks=list(callback_early_stopping(monitor='val_loss',
                                         patience=3,
                                         verbose=0,
                                         mode='auto')
                 )
)

#Save stats
test.prediction<-predict(model, as.matrix(genome.x.test))  


}


# Validation --------------------------------------------------------------

save_model_hdf5(model, "/home/nacho/DNA-RNN-main/Human.")
