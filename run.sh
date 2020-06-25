#nohup python -u train.py --model R2Plus1D --epoch 100 --batchsize 16 > logdir/r0.log  2>&1 &
nohup python -u train.py --model R2Plus1D --epoch 100 --batchsize 48 > logdir/r1.log  2>&1 &
