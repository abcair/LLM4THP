import deepchem as dc
from rdkit import Chem
import numpy as np
from transformers import T5Tokenizer, T5Model,T5EncoderModel
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer,FeatureExtractionPipeline
import torch
import esm
from transformers import AutoTokenizer
from transformers import AutoModel
import re
from Bio import SeqIO
import pickle
import argparse

aaindex1_dict = {}
aaindex1_path = "./lib/aaindex1.my.csv"
lines = open(aaindex1_path,"r").readlines()
for line in lines:
    id = line.strip().split(",")[0]
    vec = np.array([float(x) for x in  line.strip().split(",")[1:]])
    aaindex1_dict[id] = vec

def infer_aaindex1(seq):
    res = []
    for x in seq:
        tmp = aaindex1_dict[x]
        res.append(tmp)
    res = np.array(res).mean(axis=0)
    return list(res)

def infer_RDKitDescriptors(seq):
    mol = Chem.MolFromFASTA(seq)
    ss = Chem.MolToSmiles(mol)
    featurizer = dc.feat.RDKitDescriptors()
    features = featurizer.featurize(ss)
    res = features[0]
    new_res = []
    for x in res:
        if not np.isnan(x):
            new_res.append(float(x))
        else:
            new_res.append(0)
    return new_res

layer = 33
dim = 1280
model_esm2, alphabet_esm2 = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet_esm2.get_batch_converter()
def infer_esm(seq):
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("protein1", seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet_esm2.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model_esm2(batch_tokens, repr_layers=[layer], return_contacts=True)
    token_representations = results["representations"][layer].cpu()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        # sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].numpy())
    res = sequence_representations[0].mean(axis=0).tolist()
    return res



tokenizer_uniref50 = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50',do_lower_case=False)
model_uniref50 = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')

def infer_uniref50(seq):
    seq = seq
    sequences_Example = [" ".join(list(seq))]
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    ids = tokenizer_uniref50.batch_encode_plus(sequences_Example,
                                      add_special_tokens=True,
                                      padding=False)

    input_ids = torch.tensor(ids['input_ids'])
    attention_mask = torch.tensor(ids['attention_mask'])

    input_ids = input_ids
    attention_mask = attention_mask

    with torch.no_grad():
        # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)
        embedding = model_uniref50(input_ids=input_ids,
                          attention_mask=attention_mask,
                          # decoder_input_ids=input_ids,
                          )

    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding.last_hidden_state[0, :-1].detach().cpu().numpy().mean(axis=0).tolist()
    res = encoder_embedding
    return res



tokenizer_bfd = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd',do_lower_case=False)
model_bfd = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_bfd')

def infer_bfd(seq):
    seq = seq
    sequences_Example = [" ".join(list(seq))]
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    ids = tokenizer_bfd.batch_encode_plus(sequences_Example,
                                      add_special_tokens=True,
                                      padding=False)

    input_ids = torch.tensor(ids['input_ids'])
    attention_mask = torch.tensor(ids['attention_mask'])

    input_ids = input_ids
    attention_mask = attention_mask

    with torch.no_grad():
        # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)
        embedding = model_bfd(input_ids=input_ids,
                          attention_mask=attention_mask,
                          # decoder_input_ids=input_ids,
                          )

    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding.last_hidden_state[0, :-1].detach().cpu().numpy().mean(axis=0).tolist()
    res = encoder_embedding
    return res

model_pc = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
tokenizer_pc = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
bert_model_pc = FeatureExtractionPipeline(model=model_pc, tokenizer=tokenizer_pc,return_tensors=True,device=-1)

def infer_PubChem10M(seq):
    m = Chem.MolFromFASTA(seq)
    ss = Chem.MolToSmiles(m)[0:512]
    smiles = [ss]
    res = bert_model_pc(smiles)[0].cpu().numpy().squeeze().mean(axis=0).tolist()
    return res


def encode_seq_es_pu(seq):
    res = infer_esm(seq) + infer_uniref50(seq)
    return res

def encode_seq_aa_pb(seq):
    res = infer_aaindex1(seq) + infer_bfd(seq)
    return res

def encode_seq_pu_rd(seq):
    res = infer_uniref50(seq) + infer_RDKitDescriptors(seq)
    return res

def encode_seq_pb_pc(seq):
    res = infer_bfd(seq) + infer_PubChem10M(seq)
    return res

def read_fa(path):
    res = []
    ids = []
    rx = SeqIO.parse(path,format="fasta")
    for x in list(rx):
        id = str(x.id)
        seq = str(x.seq)
        ids.append(id)
        res.append(seq)
    return ids, res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path")
    parser.add_argument("--save_path")

    args = parser.parse_args()
    fasta_path = args.fasta_path
    save_path = args.save_path

    fsave = open(save_path,'w')

    ids, seqs = read_fa(fasta_path)
    es_pu = np.array([encode_seq_es_pu(seq) for seq in seqs])
    aa_pb = np.array([encode_seq_aa_pb(seq) for seq in seqs])
    pu_rd = np.array([encode_seq_pu_rd(seq) for seq in seqs])
    pb_pc = np.array([encode_seq_pb_pc(seq) for seq in seqs])

    aa_pb_models_f = open("./models/aa_pb_models.pkl","rb")
    es_pu_models_f = open("./models/es_pu_models.pkl","rb")
    pu_rd_models_f = open("./models/pu_rd_models.pkl","rb")
    pb_pc_models_f = open("./models/pb_pc_models.pkl","rb")
    lr_models_f = open("./models/lr.pkl","rb")

    aa_pb_models = pickle.load(aa_pb_models_f)
    es_pu_models = pickle.load(es_pu_models_f)
    pu_rd_models = pickle.load(pu_rd_models_f)
    pb_pc_models = pickle.load(pb_pc_models_f)
    lr_model = pickle.load(lr_models_f)


    layer1_out_aa_pb = np.concatenate([model.predict_proba(aa_pb) for model in aa_pb_models],axis=-1)
    layer1_out_es_pu = np.concatenate([model.predict_proba(es_pu) for model in es_pu_models],axis=-1)
    layer1_out_pu_rd = np.concatenate([model.predict_proba(pu_rd) for model in pu_rd_models],axis=-1)
    layer1_out_pb_pc = np.concatenate([model.predict_proba(pb_pc) for model in pb_pc_models],axis=-1)


    layer1_out_selected = np.concatenate([layer1_out_aa_pb,layer1_out_es_pu,layer1_out_pu_rd,layer1_out_pb_pc],axis=1)
    pred = lr_model.predict_proba(layer1_out_selected)[:,1]

    for i in range(pred.shape[0]):
        fsave.write(ids[i]+"\t"+seqs[i]+"\t"+str(pred[i])+"\n")
        fsave.flush()

'''
python infer_AI4ACEIP.py --fasta_path seq.fa --save_path ./res.txt
'''












