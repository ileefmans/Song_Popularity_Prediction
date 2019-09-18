setwd("/Users/ianleefmans/Desktop/Machine_Learning_Project")
setwd("/Users/ianleefmans/Desktop/Test")




install.packages("tuneR")
install.packages('oce')

num1<-list.files(pattern='*.mp3')
num2<-list.files(pattern='*.m4a')
data<-c()
for (i in 1:length(num1)) {
  assign(num1[i],tuneR::readMP3(num1[i]))
}

for (i in 1:length(num2)) {
  data<-append(data,assign(num2[i],tuneR::readMP3(num2[i])))
}


s1<-tuneR::readMP3("Many Men (Wish Death).mp3")
s1

snd<-s1@left     #extract signal

ind1<-seq(from=1, to=length(snd)/3, by=1)
ind2<-seq(from=length(snd)/3, to=length(snd)*(2/3), by=1)
ind3<-seq(from=length(snd)*(2/3), to=length(snd), by=1)
set.seed(1)
start1<-sample(ind1,1)
set.seed(1)
start2<-sample(ind2,1)
set.seed(1)
start3<-sample(ind3,1)
end1<-start1+132299
end2<-start2+132299
end3<-start3+132299
snd1<-snd[start1:end1]
snd2<-snd[start2:end2]
snd3<-snd[start3:end3]

mat1<-cbind(snd1,snd2,snd3)


ind<-seq(1:length(snd))
set.seed(1)
start<-sample(ind,1)
end<-start+132299
snd1<-snd[start:end]


dur<-length(snd1)/s1@samp.rate
dur


#### determine sample rate
for (i in seq(1:ncol(mat1))){
  

  fs<-s1@samp.rate

## demean snd

  ssnd1<-mat1[,i]-mean(mat1[,i])


  spec<-signal::specgram(x=ssnd1,n=300,Fs=fs,window=300,overlap=-150)

# discard phase info
  P<-abs(spec$S)
  P<-P/max(P)
#convert to decibels
  P<-10*log10(P)

  filename<-paste(i,".jpg",sep="")
  jpeg(filename)
  oce::imagep(x=spec$t, y=spec$f, z=t(P), col=oce::oce.colorsViridis, ylab='Frequency (Hz)', xlab='Time (s)', drawPalette = T, decimate = F)
  dev.off()
}

spec.s<-spec$S
dim(spec.s)

####################################################################
obs1<-c()
obs2<-c()
obs3<-c()
for (val in seq(1:ncol(P))){
  obs1<-append(obs1,mean(P[,val]))
  obs2<-append(obs2,sd(P[,val]))
  obs3<-append(obs3,range(P[,val])[1]-range(P[,val])[2])
}
obs<-cbind(obs1,obs2,obs3)

install.packages('magick')

plot(ssnd1,type='l',xlab='Samples',ylab='Amplitude')
