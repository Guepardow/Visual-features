for scene in C1 C2 C3 C4 C5 C6 C7
do 
for feature in bw colors hog vgg19 resnet18 densenet121 efficientnetB0 osnetAINMarket
do
for sigma in 0 5 10 20
do 
python main.py --dataset=WildTrack --scene=$scene --feature=$feature --sigma=$sigma
done
done
done