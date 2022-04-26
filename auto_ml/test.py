import h2o
import pandas as pd
import bentoml

h2o.init(max_mem_size="16G")

model_path = "../GBM_4_AutoML_1_20220420_134549"
model = h2o.load_model(model_path)

tag = bentoml.h2o.save("h2o_model", model)