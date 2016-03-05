p=0
for (( i=2 ; i<=1024 ; i=i*2))
do
  mkdir gmm$i 
  sed  -i -e 's/n_components: '$p'/n_components: '$i'/g' gmm.yaml	      
  nohup python gmm.py >gmm$i/out.txt 2>gmm$i/err.txt &
  p=$i  
done
