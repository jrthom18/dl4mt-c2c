import argparse
import sys
import os
import time
import json
import numpy 
import random
import nltk
nltk.download('punkt')

reload(sys)
sys.setdefaultencoding('utf-8')

sys.path.insert(0, "/home/jrthom18/data/dl4mt-c2c/char2char/")  # change appropriately
data_path = "/home/jrthom18/data/dl4mt-c2c/data/"               # change appropriately

import numpy
import cPickle as pkl
from mixer import *

def generate_evaluation_data():
    train_eval_source = data_path + "train_eval.source" 
    valid_eval_source = data_path + "valid_eval.source"

    train_eval_target = data_path + "train_eval.target"
    valid_eval_target = data_path + "valid_eval.target" 

    #TODO: Randomly pull 200 utterances from train.source, pull the ground-truth responses from train.target write to train_eval.source and train_eval.target
    train_sources = [line.rstrip('\n') for line in open(data_path + 'train.source')]
    train_targets = [line.rstrip('\n') for line in open(data_path + 'train.target')]
    rand_train_sources_batch = numpy.random.choice(train_sources, size=200, replace=False).tolist()
    rand_train_targets_batch = [train_targets[train_sources.index(u)] for u in rand_train_sources_batch]
    for utt in rand_train_sources_batch:
        f = open(train_eval_source, 'a')
        f.write(utt + '\n')
        f.close()
    for res in rand_train_targets_batch:
        f = open(train_eval_target, 'a')
        f.write(res + '\n')
        f.close()  

    #TODO: Randomly pull 200 utterances from valid.source, pull the ground-truth responses from valid.target write to valid_eval.source and valid_eval.target
    valid_sources = [line.rstrip('\n') for line in open(data_path + 'valid.source')]
    valid_targets = [line.rstrip('\n') for line in open(data_path + 'valid.target')]
    rand_valid_sources_batch = numpy.random.choice(valid_sources, size=200, replace=False).tolist()
    rand_valid_targets_batch = [valid_targets[valid_sources.index(u)] for u in rand_valid_sources_batch]
    for utt in rand_valid_sources_batch:
        f = open(valid_eval_source, 'a')
        f.write(utt + '\n')
        f.close()
    for res in rand_valid_targets_batch:
        f = open(valid_eval_target, 'a')
        f.write(res + '\n')
        f.close() 

    return [[train_eval_source, valid_eval_source],[train_eval_target, valid_eval_target]]

def translate_model(jobqueue, resultqueue, model, options, k, normalize, build_sampler, gen_sample, init_params, model_id, silent):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    use_noise = theano.shared(numpy.float32(0.))
    f_init, f_next = build_sampler(tparams, options, trng, use_noise)

    def _translate(seq):
        use_noise.set_value(0.)
        # sample given an input sequence and obtain scores
        # NOTE : if seq length too small, do something about it
        sample, score = gen_sample(tparams, f_init, f_next,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   options, trng=trng, k=k, maxlen=500,
                                   stochastic=False, argmax=False)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample]) 
            score = score / lengths
        sidx = numpy.argmin(score)
        return sample[sidx]

    while jobqueue:
        req = jobqueue.pop(0)

        idx, x = req[0], req[1]
        if not silent:
            print "sentence", idx, model_id
        seq = _translate(x)

        resultqueue.append((idx, seq))
    return

def main(model, dictionary, dictionary_target, sources, ground_truth, saveto, k=5,
         normalize=False, encoder_chr_level=False,
         decoder_chr_level=False, utf8=False, 
          model_id=None, silent=False):

    from char_base import (build_sampler, gen_sample, init_params)

    # load model model_options
    # /misc/kcgscratch1/ChoGroup/jasonlee/dl4mt-cdec/models/one-multiscale-conv-two-hw-lngru-1234567-100-150-200-200-200-200-200-66-one.pkl
    pkl_file = model.split('.')[0] + '.pkl'
    with open(pkl_file, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    #word_idict[0] = 'ZERO'
    #word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    #word_idict_trg[0] = 'ZERO'
    #word_idict_trg[1] = 'UNK'

    # create input and output queues for processes
    jobqueue = []
    resultqueue = []

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                if utf8:
                    ww.append(word_idict_trg[w].encode('utf-8'))
                else:
                    ww.append(word_idict_trg[w])
            if decoder_chr_level:
                capsw.append(''.join(ww))
            else:
                capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fname):
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                # idx : 0 ... len-1 
                pool_window = options['pool_stride']

                if encoder_chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()

                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)
                x = [2] + x + [3]

                # len : 77, pool_window 10 -> 3 
                # len : 80, pool_window 10 -> 0
                #rem = pool_window - ( len(x) % pool_window )
                #if rem < pool_window:
                #    x += [0]*rem

                while len(x) % pool_window != 0:
                    x += [0]

                x = [0]*pool_window + x + [0]*pool_window

                jobqueue.append((idx, x))

        return idx+1

    def _retrieve_jobs(n_samples, silent):
        trans = [None] * n_samples

        for idx in xrange(n_samples):
            resp = resultqueue.pop(0)
            trans[resp[0]] = resp[1]
            if numpy.mod(idx, 10) == 0:
                if not silent:
                    print 'Sample ', (idx+1), '/', n_samples, ' Done', model_id
        return trans

    #TODO: Measure word-for-word accuracy using NLTK tokenizer
    def _word_accuracy(gt, lr):
        ground_truth_words = nltk.word_tokenize(gt)
        lumen_response_words = nltk.word_tokenize(lr)
        
        correct_word_count = 0
        
        for inc in range(len(ground_truth_words)):
            if inc < len(lumen_response_words): 
                if lumen_response_words[inc] == ground_truth_words[inc]:
                    correct_word_count += 1

        # Calculate % accuracy measurment 
        if len(ground_truth_words) > 0:
            accuracy = correct_word_count / float(len(ground_truth_words))
        else:
            accuracy = 0
        return accuracy * 100


    #TODO: Loop thru source files
    file_iterator = 0
    training_accuracy_scores = []
    validation_accuracy_scores = []

    for source_file in sources:
        print 'Translating ', source_file, '...'
        n_samples = _send_jobs(source_file)
        print "jobs sent"

        translate_model(jobqueue, resultqueue, model, options, k, normalize, build_sampler, gen_sample, init_params, model_id, silent)
        trans = _seqs2words(_retrieve_jobs(n_samples, silent))
        print "translations retrieved"

        sf = open(source_file)
        sflines = sf.readlines()
        gtf = open(ground_truth[file_iterator])
        gtflines = gtf.readlines()

        # TODO: Write evaluations of training and validation set random samples
        with open(saveto, 'a') as f:
            if file_iterator == 0:
                f.write('TRAINING SET SAMPLES (SEEN):\n\n')

            for i in range(n_samples):
                source_input = sflines[i]
                ground_truth_response = gtflines[i]
                lumen_response = trans[i].encode('utf-8')
                
                f.write('Sample {0}\n'.format(i+1))
                f.write('Utterance: {0}'.format(source_input))
                f.write('Response:  {0}'.format(ground_truth_response))
                f.write('Lumen:     {0}\n'.format(lumen_response))
                #TODO: Measure word-for-word accuracy of responses to samples                
                accuracy_percentage = _word_accuracy(ground_truth_response, lumen_response)
                f.write('Word-for-word accuracy: {0} %\n'.format(accuracy_percentage))
                
                if file_iterator == 0:
                    training_accuracy_scores.append(accuracy_percentage)
                if file_iterator == 1:
                    validation_accuracy_scores.append(accuracy_percentage)

                f.write('---------------------------------------------------------------\n')
            
            if file_iterator == 0:
                total_accuracy = sum(training_accuracy_scores) / float(len(training_accuracy_scores))
                f.write('\n\n**********************\nTotal training set word-for-word accuracy: {0} %\n**********************\n\n'.format(total_accuracy))
                f.write('VALIDATION SET SAMPLES (UNSEEN):\n\n')
            else:
                total_accuracy = sum(validation_accuracy_scores) / float(len(validation_accuracy_scores))
                f.write('\n\n**********************\nTotal validation set word-for-word accuracy: {0} %\n**********************\n\n'.format(total_accuracy))


        file_iterator += 1
        print "Done", saveto

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=20) # beam width
    parser.add_argument('-n', action="store_true", default=True) # normalize scores for different hypothesis based on their length (to penalize shorter hypotheses, longer hypotheses are already penalized by the BLEU measure, which is precision of sorts).
    parser.add_argument('-enc_c', action="store_true", default=True) # is encoder character-level?
    parser.add_argument('-dec_c', action="store_true", default=True) # is decoder character-level?
    parser.add_argument('-utf8', action="store_true", default=True)
    parser.add_argument('-many', action="store_true", default=False) # multilingual model?
    parser.add_argument('-model', type=str) # absolute path to a model (.npz file)
    parser.add_argument('-translate', type=str, help="de_en / cs_en / fi_en / ru_en") # which language?
    parser.add_argument('-saveto', type=str, ) # absolute path where the translation should be saved
    parser.add_argument('-which', type=str, help="dev / test1 / test2", default="dev") # if you wish to translate any of development / test1 / test2 file from WMT15, simply specify which one here
    parser.add_argument('-source', type=str, default="") # if you wish to provide your own file to be translated, provide an absolute path to the file to be translated
    parser.add_argument('-silent', action="store_true", default=False) # suppress progress messages

    args = parser.parse_args()

    char_base = args.model.split("/")[-1]

    dictionary = data_path + "train.source.1004.pkl"          # change appropriately
    dictionary_target = data_path + "train.target.1002.pkl"   # change appropriately

    eval_data = generate_evaluation_data()
    source = eval_data[0]
    ground_truth = eval_data[1]

    saveto = data_path + "eval.out"    

    print "src dict:", dictionary
    print "trg dict:", dictionary_target
    print "source:", source
    print "dest:", saveto

    print args

    time1 = time.time()
    main(args.model, dictionary, dictionary_target, source, ground_truth,
         saveto, k=args.k, normalize=args.n, encoder_chr_level=args.enc_c,
         decoder_chr_level=args.dec_c,
         utf8=args.utf8,
         model_id=char_base,
         silent=args.silent,
        )
    time2 = time.time()
    duration = (time2-time1)/float(60)
    print("Translation took %.2f minutes" % duration)