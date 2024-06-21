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
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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
    rx = SeqIO.parse(path,format="fasta")
    for x in list(rx):
        seq = str(x.seq)
        res.append(seq)
    return res

'''
[(ES, PU), HGB], [(AA, PB), HGB], [(PU, RD), ET], [(PB, PC), LGB], [(AA, PB), LGB], [(ES, PU), ETS], [(ES, PU), DT], [(PB, PC), ET], [(ES, PU), ET], [(PU, RD), HGB], [(PB, PC), HGB], [(AA, PB), GB], [(ES, PU), XGB], [(PU, RD), LGB], [(AA, PB), DT], [(PU, RD), DT], [(AA, PB), ETS], [(PB, PC), ETS], [(PB, PC), DT], [(AA, PB), ET], [(ES, PU), LGB], [(PB, PC), XGB], [(PU, RD), ETS], [(PB, PC), GB], [(AA, PB), XGB], [(PU, RD), XGB], [(PU, RD), GB], [(ES, PU), GB], [(PU, RD), RF], [(ES, PU), RF], [(AA, PB), RF], [(PB, PC), RF], [(ES, PU), ADA], [(PB, PC), ADA], [(PU, RD), ADA], [(AA, PB), ADA], [(ES, PU), CAT], [(PU, RD), CAT], [(AA, PB), CAT] and [(PB, PC), CAT]
'''

neg_seqs = read_fa("../data/neg.fa")
pos_seqs = read_fa("../data/pos.fa")

es_pu_neg = [encode_seq_es_pu(seq) for seq in neg_seqs]
aa_pb_neg = [encode_seq_aa_pb(seq) for seq in neg_seqs]
pu_rd_neg = [encode_seq_pu_rd(seq) for seq in neg_seqs]
pb_pc_neg = [encode_seq_pb_pc(seq) for seq in neg_seqs]

es_pu_pos = [encode_seq_es_pu(seq) for seq in pos_seqs]
aa_pb_pos = [encode_seq_aa_pb(seq) for seq in pos_seqs]
pu_rd_pos = [encode_seq_pu_rd(seq) for seq in pos_seqs]
pb_pc_pos = [encode_seq_pb_pc(seq) for seq in pos_seqs]

aa_pb = np.concatenate([aa_pb_neg,aa_pb_pos])
es_pu = np.concatenate([es_pu_neg,es_pu_pos])
pu_rd = np.concatenate([pu_rd_neg,pu_rd_pos])
pb_pc = np.concatenate([pb_pc_neg,pb_pc_pos])
label = np.array([0]*len(neg_seqs) + [1]*len(pos_seqs))

aa_pb_models = []
es_pu_models = []
pu_rd_models = []
pb_pc_models = []

for data in [aa_pb]:
    nn = 16
    lgb_model = LGBMClassifier(n_estimators=nn, random_state=42)
    xgb_model = XGBClassifier(n_estimators=nn, random_state=42)
    cat_model = CatBoostClassifier(n_estimators=nn, random_state=42)
    ada_model = AdaBoostClassifier(n_estimators=nn, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=nn, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingClassifier(n_estimators=nn, random_state=42)
    ets_model = ExtraTreesClassifier(n_estimators=nn, random_state=42, n_jobs=-1)
    hgb_model = HistGradientBoostingClassifier(max_iter=nn, random_state=42)
    dt_model = DecisionTreeClassifier(max_depth=nn, random_state=42)
    et_model = ExtraTreeClassifier(max_depth=nn, random_state=42)
    svm_model = SVC(kernel='linear', random_state=42, probability=True)
    lr_model = LogisticRegression(random_state=42)
    knn_model = KNeighborsClassifier()
    mlp_model = MLPClassifier(solver='lbfgs', random_state=42, hidden_layer_sizes=[1024, 512, 256, 128])
    for model in [lgb_model,xgb_model,cat_model,ada_model,rf_model,gb_model,ets_model,hgb_model,dt_model,et_model]:
        model.fit(data,label)
        aa_pb_models.append(model)

for data in [es_pu]:
    nn = 16
    lgb_model = LGBMClassifier(n_estimators=nn, random_state=42)
    xgb_model = XGBClassifier(n_estimators=nn, random_state=42)
    cat_model = CatBoostClassifier(n_estimators=nn, random_state=42)
    ada_model = AdaBoostClassifier(n_estimators=nn, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=nn, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingClassifier(n_estimators=nn, random_state=42)
    ets_model = ExtraTreesClassifier(n_estimators=nn, random_state=42, n_jobs=-1)
    hgb_model = HistGradientBoostingClassifier(max_iter=nn, random_state=42)
    dt_model = DecisionTreeClassifier(max_depth=nn, random_state=42)
    et_model = ExtraTreeClassifier(max_depth=nn, random_state=42)
    svm_model = SVC(kernel='linear', random_state=42, probability=True)
    lr_model = LogisticRegression(random_state=42)
    knn_model = KNeighborsClassifier()
    mlp_model = MLPClassifier(solver='lbfgs', random_state=42, hidden_layer_sizes=[1024, 512, 256, 128])
    for model in [lgb_model,xgb_model,cat_model,ada_model,rf_model,gb_model,ets_model,hgb_model,dt_model,et_model]:
        model.fit(data,label)
        es_pu_models.append(model)

for data in [pu_rd]:
    nn = 16
    lgb_model = LGBMClassifier(n_estimators=nn, random_state=42)
    xgb_model = XGBClassifier(n_estimators=nn, random_state=42)
    cat_model = CatBoostClassifier(n_estimators=nn, random_state=42)
    ada_model = AdaBoostClassifier(n_estimators=nn, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=nn, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingClassifier(n_estimators=nn, random_state=42)
    ets_model = ExtraTreesClassifier(n_estimators=nn, random_state=42, n_jobs=-1)
    hgb_model = HistGradientBoostingClassifier(max_iter=nn, random_state=42)
    dt_model = DecisionTreeClassifier(max_depth=nn, random_state=42)
    et_model = ExtraTreeClassifier(max_depth=nn, random_state=42)
    svm_model = SVC(kernel='linear', random_state=42, probability=True)
    lr_model = LogisticRegression(random_state=42)
    knn_model = KNeighborsClassifier()
    mlp_model = MLPClassifier(solver='lbfgs', random_state=42, hidden_layer_sizes=[1024, 512, 256, 128])
    for model in [lgb_model,xgb_model,cat_model,ada_model,rf_model,gb_model,ets_model,hgb_model,dt_model,et_model]:
        model.fit(data,label)
        pu_rd_models.append(model)

for data in [pb_pc]:
    nn = 16
    lgb_model = LGBMClassifier(n_estimators=nn, random_state=42)
    xgb_model = XGBClassifier(n_estimators=nn, random_state=42)
    cat_model = CatBoostClassifier(n_estimators=nn, random_state=42)
    ada_model = AdaBoostClassifier(n_estimators=nn, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=nn, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingClassifier(n_estimators=nn, random_state=42)
    ets_model = ExtraTreesClassifier(n_estimators=nn, random_state=42, n_jobs=-1)
    hgb_model = HistGradientBoostingClassifier(max_iter=nn, random_state=42)
    dt_model = DecisionTreeClassifier(max_depth=nn, random_state=42)
    et_model = ExtraTreeClassifier(max_depth=nn, random_state=42)
    svm_model = SVC(kernel='linear', random_state=42, probability=True)
    lr_model = LogisticRegression(random_state=42)
    knn_model = KNeighborsClassifier()
    mlp_model = MLPClassifier(solver='lbfgs', random_state=42, hidden_layer_sizes=[1024, 512, 256, 128])
    for model in [lgb_model,xgb_model,cat_model,ada_model,rf_model,gb_model,ets_model,hgb_model,dt_model,et_model]:
        model.fit(data,label)
        pb_pc_models.append(model)

'''
aa_pb (10, 1577)
es_pu (10, 2304)
pu_rd (10, 1234)
pb_pc (10, 1792)
aa_pb_models 10
es_pu_models 10
pu_rd_models 10
pb_pc_models 10
ValueError: Number of features of the model must match the input. Model n_features_ is 1792 and input n_features is 1577
'''

import pickle

aa_pb_model_f = open("./models/aa_pb_models.pkl","wb")
es_pu_model_f = open("./models/es_pu_models.pkl","wb")
pu_rd_model_f = open("./models/pu_rd_models.pkl","wb")
pb_pc_model_f = open("./models/pb_pc_models.pkl","wb")

pickle.dump(aa_pb_models,aa_pb_model_f)
pickle.dump(es_pu_models,es_pu_model_f)
pickle.dump(pu_rd_models,pu_rd_model_f)
pickle.dump(pb_pc_models,pb_pc_model_f)


print("aa_pb",aa_pb.shape)
print("es_pu",es_pu.shape)
print("pu_rd",pu_rd.shape)
print("pb_pc",pb_pc.shape)

print("aa_pb_models",len(aa_pb_models))
print("es_pu_models",len(es_pu_models))
print("pu_rd_models",len(pu_rd_models))
print("pb_pc_models",len(pb_pc_models))

layer1_out_aa_pb = np.concatenate([model.predict_proba(aa_pb) for model in aa_pb_models],axis=-1)
layer1_out_es_pu = np.concatenate([model.predict_proba(es_pu) for model in es_pu_models],axis=-1)
layer1_out_pu_rd = np.concatenate([model.predict_proba(pu_rd) for model in pu_rd_models],axis=-1)
layer1_out_pb_pc = np.concatenate([model.predict_proba(pb_pc) for model in pb_pc_models],axis=-1)

layer1_out_selected = np.concatenate([layer1_out_aa_pb,layer1_out_es_pu,layer1_out_pu_rd,layer1_out_pb_pc],axis=1)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42)
lr.fit(layer1_out_selected,label)

lr_f = open("./models/lr.pkl","wb")
pickle.dump(lr,lr_f)












