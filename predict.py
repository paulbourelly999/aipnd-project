import model_utils
import argparse
import json

parser = argparse.ArgumentParser(description='Python script to load a trained neural network and us it to classify and image!')
parser.add_argument('input', action='store')

parser.add_argument('model_path', action='store')
parser.add_argument('--top_k', action='store', dest='top_k', default=5)
parser.add_argument('--category_names', action='store', dest='category_names', default='./cat_to_name.json')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='use_gpu',
                    help='Use GPU')
args = parser.parse_args()
model_utils.set_device(args.use_gpu)

# Load trained model
model, optmizer, epoch, loss = model_utils.load_model(args.model_path)
top_prob, top_class = model_utils.predict(args.input, model, args.top_k,)
model_utils.log_prediction(top_prob, top_class,  args.category_names)
