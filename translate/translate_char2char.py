import argparse
import sys
import os
import time
import json

import requests
from flask import Flask, request

reload(sys)
sys.setdefaultencoding('utf-8')

sys.path.insert(0, "/home/jrthom18/data/char_model/dl4mt-c2c/char2char/") # change appropriately

import numpy
import cPickle as pkl
from mixer import *

# Globals for init_translation_model
trng = 0
tparams = 0
use_noise = 0
f_init = 0
f_next = 0

def init_translation_model(model, options, init_params, build_sampler):
    global trng
    global tparams
    global use_noise
    global f_init
    global f_next

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


def translate_model(jobqueue, resultqueue, model, options, k, normalize, build_sampler, gen_sample, init_params, model_id, silent):

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

class Translator(object):

    def __init__(self, model, dictionary, dictionary_target, source_file, saveto, k,
         normalize, encoder_chr_level,
         decoder_chr_level, utf8, 
          model_id, silent, interactive):

        self.model = model
        self.dictionary = dictionary
        self.dictionary_target = dictionary_target
        self.source_file = source_file
        self.saveto = saveto
        self.k = k
        self.normalize = normalize
        self.encoder_chr_level = encoder_chr_level
        self.decoder_chr_level = decoder_chr_level
        self.utf8 = utf8
        self.model_id = model_id
        self.silent = silent
        self.interactive = interactive

        from char_base import (build_sampler, gen_sample, init_params)

        # load model model_options
        # /misc/kcgscratch1/ChoGroup/jasonlee/dl4mt-cdec/models/one-multiscale-conv-two-hw-lngru-1234567-100-150-200-200-200-200-200-66-one.pkl
        pkl_file = self.model.split('.')[0] + '.pkl'
        with open(pkl_file, 'rb') as f:
            self.options = pkl.load(f)

        # load source dictionary and invert
        with open(self.dictionary, 'rb') as f:
            self.word_dict = pkl.load(f)
        word_idict = dict()
        for kk, vv in self.word_dict.iteritems():
            word_idict[vv] = kk
        #word_idict[0] = 'ZERO'
        #word_idict[1] = 'UNK'

        # load target dictionary and invert
        with open(self.dictionary_target, 'rb') as f:
            self.word_dict_trg = pkl.load(f)
        self.word_idict_trg = dict()
        for kk, vv in self.word_dict_trg.iteritems():
            self.word_idict_trg[vv] = kk
        #word_idict_trg[0] = 'ZERO'
        #word_idict_trg[1] = 'UNK'

        # create input and output queues for processes
        self.jobqueue = []
        self.resultqueue = []

        init_translation_model(self.model, self.options, init_params, build_sampler)
        print("Model initiation complete...ready for input.")

    def seqs2words(self, caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                if self.utf8:
                    ww.append(self.word_idict_trg[w].encode('utf-8'))
                else:
                    ww.append(self.word_idict_trg[w])
            if self.decoder_chr_level:
                capsw.append(''.join(ww))
            else:
                capsw.append(' '.join(ww))
        return capsw

    def send_jobs(self, fname):
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                # idx : 0 ... len-1 
                pool_window = self.options['pool_stride']

                if self.encoder_chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()

                x = map(lambda w: self.word_dict[w] if w in self.word_dict else 1, words)
                x = map(lambda ii: ii if ii < self.options['n_words_src'] else 1, x)
                x = [2] + x + [3]

                # len : 77, pool_window 10 -> 3 
                # len : 80, pool_window 10 -> 0
                #rem = pool_window - ( len(x) % pool_window )
                #if rem < pool_window:
                #    x += [0]*rem

                while len(x) % pool_window != 0:
                    x += [0]

                x = [0]*pool_window + x + [0]*pool_window

                self.jobqueue.append((idx, x))

        return idx+1

    def send_job(self, sentence):
        pool_window = self.options['pool_stride']

        if self.encoder_chr_level:
            words = list(sentence.decode('utf-8').strip())
        else:
            words = sentence.strip().split()

        x = map(lambda w: self.word_dict[w] if w in self.word_dict else 1, words)
        x = map(lambda ii: ii if ii < self.options['n_words_src'] else 1, x)
        x = [2] + x + [3]

        while len(x) % pool_window != 0:
            x += [0]

        x = [0]*pool_window + x + [0]*pool_window
        self.jobqueue.append((0, x))
        return 1

    def retrieve_jobs(self, n_samples, silent):
        trans = [None] * n_samples

        for idx in xrange(n_samples):
            resp = self.resultqueue.pop(0)
            trans[resp[0]] = resp[1]
            if numpy.mod(idx, 10) == 0:
                if not silent:
                    print 'Sample ', (idx+1), '/', n_samples, ' Done', self.model_id
        return trans

    def translate(self, message_text):
        from char_base import (build_sampler, gen_sample, init_params)
        n_samples = self.send_job(message_text)
        translate_model(self.jobqueue, self.resultqueue, self.model, self.options, self.k, self.normalize, build_sampler, gen_sample, init_params, self.model_id, True)
        trans = self.seqs2words(self.retrieve_jobs(n_samples, True))
        message = u' '.join(trans).encode('utf-8')
        return message

    '''
    if interactive:
        sys.stdout.write("> ")
        sys.stdout.flush()
        utterance = sys.stdin.readline()
        while utterance:
            n_samples = _send_job(utterance)
            translate_model(jobqueue, resultqueue, model, options, k, normalize, build_sampler, gen_sample, init_params, model_id, True)
            trans = _seqs2words(_retrieve_jobs(n_samples, True))
            print(u' '.join(trans).encode('utf-8'))
            sys.stdout.write("> ")
            sys.stdout.flush()
            utterance = sys.stdin.readline()

    else:
        print 'Translating ', source_file, '...'
        n_samples = _send_jobs(source_file)
        print "jobs sent"

        translate_model(jobqueue, resultqueue, model, options, k, normalize, build_sampler, gen_sample, init_params, model_id, silent)
        trans = _seqs2words(_retrieve_jobs(n_samples, silent))
        print "translations retrieved"

        with open(saveto, 'w') as f:
            print >>f, u'\n'.join(trans).encode('utf-8')

        print "Done", saveto
    '''

app = Flask(__name__)

data_path = "/home/jrthom18/data/char_model/dl4mt-c2c/data/"
model = "models/bi-char2char.grads.194124.npz"
char_base = model.split("/")[-1]
dictionary = data_path + "train.source.124.pkl"          # change appropriately
dictionary_target = data_path + "train.target.122.pkl"   # change appropriately
source = data_path + "dev.source" 
saveto = data_path + "dev.out"
k = 20

translator = Translator(model, dictionary, dictionary_target, source, saveto, k, True, True, True, True, char_base, False, True)

# Facebook Messenger app verification
VERIFY_TOKEN = "my_voice_is_my_password_verify_me"
PAGE_ACCESS_TOKEN = "EAAUCWBAyvEkBAM5kmECcUmqxH15u8bKbGuhmJ1WNXZBDZBfOPlGESoPZC8KQvKqa9oelFuZCQ3REcCLxfX5tiqTRKZCWEfnAszhlCLZCAgVWZCmZA78IW0wj9yTe3fH46Qq0mcD0FZB4wfsydRg0cB4dAu6DY5HCE69ZAJ14JY4EFZCZAgZDZD"

@app.route('/', methods=['GET'])
def verify():
    # when the endpoint is registered as a webhook, it must echo back
    # the 'hub.challenge' value it receives in the query arguments
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == VERIFY_TOKEN:
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200

    return "Hello world", 200


@app.route('/', methods=['POST'])
def webhook():
    # endpoint for processing incoming messaging events

    data = request.get_json()
    #log(data)  # you may not want to log every incoming message in production, but it's good for testing

    if data["object"] == "page":

        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:

                if messaging_event.get("message"):  # someone sent us a message

                    sender_id = messaging_event["sender"]["id"]        # the facebook ID of the person sending you the message
                    recipient_id = messaging_event["recipient"]["id"]  # the recipient's ID, which should be your page's facebook ID
                    message_text = messaging_event["message"]["text"]  # the message's text
                    post_message_text(message_text)
                    # Respond by decoding message_text
                    #handle_typing_bubble(sender_id, True)
                    message = translator.translate(message_text)
                    post_bot_response(message)
                    #handle_typing_bubble(sender_id, False)
                    #send_message(sender_id, message)

                if messaging_event.get("delivery"):  # delivery confirmation
                    pass

                if messaging_event.get("optin"):  # optin confirmation
                    pass

                if messaging_event.get("postback"):  # user clicked/tapped "postback" button in earlier message
                    pass

    return "ok", 200


def send_message(recipient_id, message_text):

    log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    params = {
        "access_token": PAGE_ACCESS_TOKEN
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text
        }
    })
    r = requests.post("https://graph.facebook.com/v2.6/me/messages", params=params, headers=headers, data=data)
    if r.status_code != 200:
        log(r.status_code)
        log(r.text)

def handle_typing_bubble(recipient_id, show):
    params = {
        "access_token": PAGE_ACCESS_TOKEN
    }
    headers = {
        "Content-Type": "application/json"
    }
    if show:
        data = json.dumps({
            "recipient": {
                "id": recipient_id
            },
            "sender_action":"typing_on"
        })
    else:
        data = json.dumps({
            "recipient": {
                "id": recipient_id
            },
            "sender_action":"typing_off"
        })

    r = requests.post("https://graph.facebook.com/v2.6/me/messages", params=params, headers=headers, data=data)

def post_message_text(message_text):
    data = { "message_text": message_text }
    r = requests.post("http://128.199.126.187/receive_message.php", data=data)

def post_bot_response(message):
    data = { "message_text": message }
    r = requests.post("http://128.199.126.187/receive_bot_response.php", data=data)    

def log(message):  # simple wrapper for logging to stdout on heroku
    print str(message)
    sys.stdout.flush()   

if __name__ == "__main__":
    app.run(debug=True)
    '''
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
    parser.add_argument('-interactive', action="store_true", default=True) # decode from command line input

    args = parser.parse_args()

    data_path = "/home/jrthom18/data/char_model/dl4mt-c2c/data/"       # change appropriately

    '''
    '''
    which_wmt = None
    if args.many:
        which_wmt = "multi-wmt15"
    else:
        which_wmt = "wmt15"

    if args.which not in "dev test1 test2".split():
        raise Exception('1')

    if args.translate not in ["de_en", "cs_en", "fi_en", "ru_en"]:
        raise Exception('1')

    if args.translate == "fi_en" and args.which == "test2":
        raise Exception('1')

    if args.many:
        from wmt_path_iso9 import *

        dictionary = wmts['many_en']['dic'][0][0]
        dictionary_target = wmts['many_en']['dic'][0][1]
        source = wmts[args.translate][args.which][0][0]

    else:
        from wmt_path import *

        aa = args.translate.split("_")
        lang = aa[0]
        en = aa[1]

        dictionary = "%s%s/train/all_%s-%s.%s.tok.304.pkl" % (lang, en, lang, en, lang)
        dictionary_target = "%s%s/train/all_%s-%s.%s.tok.300.pkl" % (lang, en, lang, en, en)
        source = wmts[args.translate][args.which][0][0]
    '''
    '''
    char_base = args.model.split("/")[-1]

    dictionary = data_path + "train.source.124.pkl"          # change appropriately
    dictionary_target = data_path + "train.target.122.pkl"   # change appropriately
    source = data_path + "dev.source" 

    if args.source != "":
        source = args.source

    print "src dict:", dictionary
    print "trg dict:", dictionary_target
    print "source:", source
    print "dest :", args.saveto

    print args

    time1 = time.time()
    main(args.model, dictionary, dictionary_target, source,
         args.saveto, k=args.k, normalize=args.n, encoder_chr_level=args.enc_c,
         decoder_chr_level=args.dec_c,
         utf8=args.utf8,
         model_id=char_base,
         silent=args.silent,
         interactive=args.interactive,
        )
    time2 = time.time()
    duration = (time2-time1)/float(60)
    print("Translation took %.2f minutes" % duration)
    '''