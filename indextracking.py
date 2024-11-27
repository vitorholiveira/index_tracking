import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Dados fictícios
np.random.seed(42)  # Para reprodutibilidade
I = range(5)  # Conjunto de ativos disponíveis
T = range(10)  # Número de períodos
K = 3  # Número máximo de ativos permitidos
r = {(t, i): np.random.uniform(-0.02, 0.05) for t in T for i in I}  # Rendimento dos ativos
R = [np.random.uniform(-0.01, 0.04) for t in T]  # Rendimento do índice


def get_portfolio(I,T,K,r,R):
    # Modelo
    model = gp.Model("TrackingPortfolio")
    w = model.addVars(I, lb=0, ub=1, name="w")  # Pesos dos ativos
    z = model.addVars(I, vtype=GRB.BINARY, name="z")  # Seleção de ativos

    # Função objetivo
    model.setObjective(
        (1 / len(T)) * gp.quicksum(
            (gp.quicksum(w[i] * r[t, i] for i in I) - R[t]) ** 2 for t in T
        ),
        GRB.MINIMIZE
    )

    # Restrições
    model.addConstr(gp.quicksum(w[i] for i in I) == 1, "SomaPesos")  # Soma dos pesos deve ser 1
    model.addConstrs((w[i] <= z[i] for i in I), "VinculoPesos")      # Pesos vinculados a z
    model.addConstr(gp.quicksum(z[i] for i in I) <= K, "MaxAtivos")  # Máximo de K ativos

    # Resolver
    model.optimize()

    # Resultados
    if model.Status == GRB.OPTIMAL:
        print("\nSolução ótima encontrada!")
        for i in I:
            print(f"Ativo {i}: Peso = {w[i].x:.4f}, Selecionado = {int(z[i].x)}")
    else:
        print("\nNenhuma solução ótima encontrada.")

    portfolio_weights = [w[i].x for i in I]
    
    return portfolio_weights

portfolio_weights = get_portfolio(I,T,K,r,R)