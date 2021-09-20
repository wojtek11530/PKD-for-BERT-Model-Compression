import argparse
import logging
import math
import os

import numpy as np
import torch
from thop import profile

from BERT.pytorch_pretrained_bert.modeling import BertConfig
from BERT.pytorch_pretrained_bert.tokenization import BertTokenizer
from src.data_processing import init_model
from src.modeling import BertForSequenceClassificationEncoder, FullFCClassifierForSequenceClassification
from src.nli_data_processing import processors
from src.utils import load_model, count_parameters
from envs import HOME_DATA_FOLDER

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


#########################################################################
# Prepare Parser
##########################################################################

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_parser():
    parser = argparse.ArgumentParser()
    # Input Training tasks
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        help="The name of the task for training.")

    parser.add_argument('--student_hidden_layers',
                        type=int,
                        default=None,
                        help="number of transformer layers for student, default is None (use all layers)")

    parser.add_argument("--kd_model",
                        default="kd",
                        type=str,
                        help="KD model architecture, either kd, kd.full or kd.cls")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--bert_model",
                        default=None,
                        type=str,
                        help="student bert model configuration folder")

    parser.add_argument("--encoder_checkpoint",
                        default=None,
                        type=str,
                        help="check point for student encoder")

    parser.add_argument("--cls_checkpoint",
                        default=None,
                        type=str,
                        help="check point for student classifier")

    parser.add_argument('--fp16',
                        type=boolean_string,
                        default=False,
                        help="Whether to use 16-bit float precision instead of 32-bit")

    return parser


def complete_argument(args):
    MODEL_FOLDER = os.path.join(HOME_DATA_FOLDER, 'models')
    if args.student_hidden_layers in [None, 'None']:
        args.student_hidden_layers = 12 if 'base' in args.bert_model else 24
    args.bert_model = os.path.join(MODEL_FOLDER, 'pretrained', args.bert_model)

    if args.encoder_checkpoint not in [None, 'None']:
        args.encoder_checkpoint = os.path.join(MODEL_FOLDER, args.encoder_checkpoint)
    else:
        args.encoder_checkpoint = os.path.join(MODEL_FOLDER, 'pretrained', args.bert_model, 'pytorch_model.bin')
        logger.info('encoder checkpoint not provided, use pre-trained at %s instead' % args.encoder_checkpoint)
    if args.cls_checkpoint not in [None, 'None']:
        args.cls_checkpoint = os.path.join(MODEL_FOLDER, args.cls_checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    args.device = device
    args.n_gpu = n_gpu

    return args


parser = get_parser()
args = parser.parse_args()
args = complete_argument(args)

task_name = args.task_name.lower()

if task_name not in processors and 'race' not in task_name:
    raise ValueError("Task not found: %s" % task_name)

if 'race' in task_name:
    pass
else:
    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

#########################################################################
# Prepare model
#########################################################################
student_config = BertConfig(os.path.join(args.bert_model, 'bert_config.json'))
if args.kd_model.lower() in ['kd', 'kd.cls']:
    logger.info('using normal Knowledge Distillation')
    output_all_layers = args.kd_model.lower() == 'kd.cls'
    student_encoder, student_classifier = init_model(task_name, output_all_layers, args.student_hidden_layers,
                                                     student_config)

    n_student_layer = len(student_encoder.bert.encoder.layer)
    student_encoder = load_model(student_encoder, args.encoder_checkpoint, args, 'student', verbose=True)
    logger.info('*' * 77)
    student_classifier = load_model(student_classifier, args.cls_checkpoint, args, 'classifier', verbose=True)
elif args.kd_model.lower() == 'kd.full':
    logger.info('using FULL Knowledge Distillation')
    layer_idx = [int(i) for i in args.fc_layer_idx.split(',')]
    num_fc_layer = len(layer_idx)
    if args.weights is None or args.weights.lower() in ['none']:
        weights = np.array([1] * (num_fc_layer - 1) + [num_fc_layer - 1]) / 2 / (num_fc_layer - 1)
    else:
        weights = [float(w) for w in args.weights.split(',')]
        weights = np.array(weights) / sum(weights)

    assert len(weights) == num_fc_layer, 'number of weights and number of FC layer must be equal to each other'

    # weights = torch.tensor(np.array([1, 1, 1, 1, 2, 6])/12, dtype=torch.float, device=device, requires_grad=False)
    # if args.fp16:
    #    weights = weights.half()
    student_encoder = BertForSequenceClassificationEncoder(student_config, output_all_encoded_layers=True,
                                                           num_hidden_layers=args.student_hidden_layers,
                                                           fix_pooler=True)
    n_student_layer = len(student_encoder.bert.encoder.layer)
    student_encoder = load_model(student_encoder, args.encoder_checkpoint, args, 'student', verbose=True)
    logger.info('*' * 77)

    student_classifier = FullFCClassifierForSequenceClassification(student_config, num_labels,
                                                                   student_config.hidden_size,
                                                                   student_config.hidden_size, 6)
    student_classifier = load_model(student_classifier, args.cls_checkpoint, args, 'exact', verbose=True)
    assert max(layer_idx) <= n_student_layer - 1, 'selected FC layer idx cannot exceed the number of transformers'
else:
    raise ValueError('%s KD not found, please use kd or kd.full' % args.kd)

n_param_student = count_parameters(student_encoder) + count_parameters(student_classifier)
logger.info('number of layers in student model = %d' % n_student_layer)
logger.info('num parameters in student model are %d and %d' % (
    count_parameters(student_encoder), count_parameters(student_classifier)))

encoder_input = tuple([torch.randint(high=len(tokenizer.vocab),
                                     size=(1, args.max_seq_length), dtype=torch.int64, device=args.device),
                       torch.randint(high=1, size=(1, args.max_seq_length), dtype=torch.int64, device=args.device),
                       torch.randint(high=1, size=(1, args.max_seq_length), dtype=torch.int64, device=args.device)])

encoder_macs, encoder_params = profile(student_encoder, inputs=encoder_input)

cls_input = torch.randn(1, student_config.hidden_size, device=args.device)
cls_macs, cls_params = profile(student_classifier, inputs=(cls_input,))


def print_results(macs, params, title=''):
    if len(title) != 0:
        print("- " + title)
    print(f"\tmacs [G]: {macs / math.pow(10, 9):.2f}, params [M]: {params / math.pow(10, 6):.2f}")


print("Results")
print_results(encoder_macs, encoder_params, 'Encoder')
print_results(cls_macs, cls_params, 'Classifier')
print_results(encoder_macs + cls_macs, encoder_params + cls_params, 'Whole model')
