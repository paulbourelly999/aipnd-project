import model_utils
import argparse
import json


parser = argparse.ArgumentParser(description='Python script to train a VGG or DenseNet Neural Network for Image Recoginition!')
parser.add_argument('--save_dir', action='store', dest='save_dir', default='./models/', help='Directory to save trained Neural Network to.')
parser.add_argument('--arch', action='store', dest='arch', default='densenet_121', help='Neural Network architecture')
parser.add_argument('--learn_rate', action='store', dest='learn_rate', type=float, default='0.001', help='Learn rate for classifier training.')
parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default='500', help='Number of units in hidden layer of classifier.')
parser.add_argument('--epochs', action='store', dest='epochs', type=int, default='5', help='Number of Epochs to train for')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='use_gpu',
                    help='Use GPU')
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
args = parser.parse_args()
model_utils.set_device(args.use_gpu)

model, optimizer = model_utils.create_model(args.arch, args.hidden_units, args.learn_rate)
data = model_utils.load_data("flowers/train/", model)
model,optimizer,epoch,loss = model_utils.train_model(model,data,args.epochs, optimizer)
model_utils.save_model(model,args.hidden_units, optimizer,args.arch,args.learn_rate, args.save_dir,epoch, loss)

