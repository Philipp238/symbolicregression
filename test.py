import torch
import numpy as np
import sympy as sp
import os, sys
import symbolicregression
import requests
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model_path = "model.pt" 
    try:
        if not os.path.isfile(model_path): 
            url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)
        if not torch.cuda.is_available():
            model = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            model = torch.load(model_path)
            model = model.cuda()
        print(model.device)
        print("Model successfully loaded!")

    except Exception as e:
        print("ERROR: model not loaded! path was: {}".format(model_path))
        print(e)    

    est = symbolicregression.model.SymbolicTransformerRegressor(
                        model=model,
                        max_input_points=200,
                        n_trees_to_refine=100,
                        rescale=True
                        )

    ## Example of data
    dim = 2
    x = np.random.randn(100, dim)
    y = np.cos(2*np.pi*x[:,0]) +x[:,1]**2

    
    est.fit(x,y)
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    model_str = est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()
    for op,replace_op in replace_ops.items():
        model_str = model_str.replace(op,replace_op)
    print(sp.parse_expr(model_str))
    
    y_pred = est.predict(x)
    print(f'L2 distance: {np.linalg.norm(y_pred - y) / len(y)}')

    if dim == 1:
        index = np.argsort(x[:,0])
        plt.plot(x[index], y[index])
        plt.plot(x[index], y_pred[index])
        plt.show()