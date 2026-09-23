"""Frozen architecture reconstruction and fitted JSON prediction adapter."""
import json
from pathlib import Path
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from portable_predict import predict_proba

def build_unfitted(spec):
    blocks=[]
    for b in spec['preprocessing']:
        params=dict(b['imputer_parameters']);params['missing_values']=np.nan
        steps=[('imputer',SimpleImputer(**params))]
        if b['kind']=='numeric':
            if b.get('scale') is not None:steps.append(('scaler',StandardScaler()))
            name='numeric'
        else:
            assert all(x is None for x in b['drop_indices'])
            steps.append(('onehot',OneHotEncoder(handle_unknown=b['handle_unknown'],sparse_output=False)))
            name='categorical'
        blocks.append((name,Pipeline(steps),b['columns']))
    cls=LogisticRegression if spec['estimator']['kind']=='logistic' else HistGradientBoostingClassifier
    return Pipeline([('preprocess',ColumnTransformer(blocks)),('model',cls(**spec['estimator']['parameters']))])

class FrozenJSONModel:
    def __init__(self,path):
        self.path=Path(path);self.spec=json.loads(self.path.read_text())
        self.feature_names_in_=np.array(self.spec['ordered_input_features'],dtype=object)
        self.classes_=np.array(self.spec['estimator']['classes'])
    def predict_proba(self,frame):return predict_proba(frame,self.spec)
    def __sklearn_clone__(self):return build_unfitted(self.spec)

def source_bundle(parameters,label):
    model=FrozenJSONModel(Path(parameters)/f'mimic_{label}__dynamic_multinomial_logistic.json')
    return {'models':{'dynamic_multinomial_logistic':model},'features':{'dynamic_multinomial_logistic':list(model.feature_names_in_)}}

def ablation_bundle(parameters):
    return {p.stem.removeprefix('ablation__'):FrozenJSONModel(p) for p in Path(parameters).glob('ablation__*.json')}
