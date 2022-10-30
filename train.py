import model_utils
import argparse
import json


parser = argparse.ArgumentParser(description='Python script to train a VGG or DenseNet Neural Network for Image Recoginition!')
parser.add_argument('--save_dir', action='store', dest='save_dir', default='./models')
parser.add_argument('--arch', action='store', dest='arch', default='densenet_121')
parser.add_argument('--learn_rate', action='store', dest='learn_rate', type=float, default='0.1')
parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default='500')
parser.add_argument('--epochs', action='store', dest='epochs', type=int, default='10')

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
args = parser.parse_args()
model, optimizer = model_utils.create_model(args.arch, args.hidden_units, args.learn_rate, cat_to_name)
data = model_utils.load_data("flowers/train/")
model,optimizer,epoch,loss = model_utils.train_model(model,data,args.epochs, optimizer)
val_data = model_utils.load_val_data()
model_utils.model_eval(model, val_data)
model_utils.save_model(model, optimizer,args.arch,args.learn_rate, args.save_dir,epoch, loss)
print(args)
