import argparse
import time
import math
import torch
import torch.nn.functional as F
from torch.distributions import LogNormal, Dirichlet
from torch.distributions import kl_divergence

from prodLDA import ProdLDA
import data


parser = argparse.ArgumentParser(description='ProdLDA')
parser.add_argument('--hidden_size', type=int, default=256,
                    help="number of hidden units for hidden layers")
parser.add_argument('--num_topics', type=int, default=16,
                    help="number of topics")
parser.add_argument('--dropout', type=float, default=0.2,
                    help="dropout applied to layers (0 = no dropout)")
parser.add_argument('--data', type=str, default='./data/20news',
                    help="location of the data folder")
parser.add_argument('--use_lognormal', action='store_true',
                    help="Use LogNormal to approximate Dirichlet")
parser.add_argument('--epochs', type=int, default=5,
                    help="maximum training epochs")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help="batch size")
parser.add_argument('--lr', type=float, default=1e-3,
                    help="learning rate")
parser.add_argument('--wd', type=float, default=0,
                    help="weight decay used for regularization")
parser.add_argument('--epoch_size', type=int, default=2000,
                    help="number of training steps in an epoch")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
parser.add_argument('--nocuda', action='store_true',
                    help="do not use CUDA")
args = parser.parse_args()

torch.manual_seed(args.seed)





def main(args):
    print("Loading data")
    dataset = args.data.rstrip('/').split('/')[-1]
    corpus = data.Corpus(args.data)
    vocab_size = len(corpus.vocab)
    print("\ttraining data size: ", corpus.train.size)
    print("\tvocabulary size: ", vocab_size)
    print("Constructing model")
    print(args)
    device = torch.device('cpu' if args.nocuda else 'cuda')
    model = ProdLDA(
        None,vocab_size, args.hidden_size, args.num_topics,
        args.dropout, args.use_lognormal)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_loss = None

    print("\nStart training")
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_nll, train_kld, train_ppl = model._train(args, corpus.train, model, optimizer, epoch, device)
            test_nll, test_kld, test_ppl = model._evaluate(args, corpus.test, model, device)
            print('-' * 80)
            meta = "| epoch {:2d} | time {:5.2f}s ".format(epoch, time.time()-epoch_start_time)
            print(meta + "| train loss {:5.2f} ({:4.2f}) "
                  "| train ppl {:5.2f}".format(
                      train_nll, train_kld, train_ppl))
            print(len(meta) * ' ' + "| test loss  {:5.2f} ({:4.2f}) "
                  "| test ppl  {:5.2f}".format(
                      test_nll, test_kld, test_ppl), flush=True)
            if best_loss is None or test_nll + test_kld < best_loss:
                best_loss = test_nll + test_kld
                with open(model.get_savepath(args), 'wb') as f:
                    torch.save(model, f)
                
    except KeyboardInterrupt:
        print('-' * 80)
        print('Exiting from training early')


    with open(model.get_savepath(args), 'rb') as f:
        model = torch.load(f)

    test_nll, test_kld, test_ppl = model._evaluate(args,corpus.test, model, device)
    print('=' * 80)
    print("| End of training | test loss {:5.2f} ({:5.2f}) "
          "| test ppl {:5.2f}".format(
              test_nll, test_kld, test_ppl))
    print('=' * 80)
    emb = model.beta()
    idx2word = dict((i, w) for (w, i) in corpus.vocab.items())
    model.print_top_words(emb, idx2word)

    
if __name__ == '__main__':
    main(args)
