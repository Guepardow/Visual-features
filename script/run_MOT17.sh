for scene in 02 04 05 09 10 11 13  
do 
for feature in bw colors hog vgg19 resnet18 densenet121 efficientB0 osnetAINMarket
do
for sigma in 0 5 10 20
do 
python main.py --dataset=MOT17 --scene=$scene --feature=$feature --sigma=$sigma
done
done
done