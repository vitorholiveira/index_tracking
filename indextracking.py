import numpy as np
import gurobipy as gp
from gurobipy import GRB

class IndexTracking:
    # Portfolio Configuration Constants
    DEFAULT_PORTFOLIO_SIZE = 15
    DEFAULT_TRAIN_RATIO = 0.7
    DEFAULT_MAX_ITERATIONS = 100000
    DEFAULT_MULTI_START_ITERATIONS = 10

    def __init__(self, stock_data, index_data):
        """
        Initialize the IndexTracking class
        
        Parameters:
        - stock_data: Full DataFrame with stock returns
        - index_data: Full array or Series with index returns
        """
        self.full_stock_data = stock_data
        self.full_index_data = index_data
        
        # Default train-test split
        self.train_stock_data = None
        self.train_index_data = None
        self.test_stock_data = None
        self.test_index_data = None
        self.best_result = None
        self.results = None
        self.best_result_is = None
        self.results_is = None
    
    def split_data(self, train_start, train_end, test_start, test_end):
        """
        Split data into training and testing sets
        
        Parameters:
        - train_ratio: Proportion of data to use for training (default 70%)
        """
        self.train_stock_data = self.full_stock_data.loc[train_start:train_end]
        self.test_stock_data = self.full_stock_data.loc[test_start:test_end]

        self.train_index_data = self.full_index_data.loc[train_start:train_end]
        self.test_index_data = self.full_index_data.loc[test_start:test_end]
        
        return {
            'train_periods': len(self.train_stock_data),
            'test_periods': len(self.test_stock_data)
        }
    
    def create_portfolio(self, 
                         portfolio_size=DEFAULT_PORTFOLIO_SIZE, 
                         max_iterations=DEFAULT_MAX_ITERATIONS,
                         initial_solution=None):
        """
        Create an optimized portfolio that tracks the index
        
        Parameters:
        - portfolio_size: Number of stocks to select
        - max_iterations: Maximum Gurobi iterations
        
        Returns:
        - Dictionary with portfolio weights and tracking error
        """
        if self.train_stock_data is None:
            print('Error')
            return

        stock_data = self.train_stock_data
        index_data = self.train_index_data

        I = np.array(stock_data.columns)
        T = range(len(stock_data))
        r = {(t, i): stock_data.iloc[t][i] for t in T for i in I}
        R = np.array(index_data)
        k = portfolio_size

        model = gp.Model("TrackingPortfolio")
        w = model.addVars(I, lb=0, ub=1, name="w")  # Stock weights
        z = model.addVars(I, vtype=GRB.BINARY, name="z")  # Stock selection

        model.setObjective(
            (1 / len(T)) * gp.quicksum(
                (gp.quicksum(w[i] * r[t, i] for i in I) - R[t]) ** 2 for t in T
            ),
            GRB.MINIMIZE
        )

        model.addConstr(gp.quicksum(w[i] for i in I) == 1, "WeightSum")
        model.addConstrs((w[i] <= z[i] for i in I), "WeightBinding")
        model.addConstr(gp.quicksum(z[i] for i in I) == k, "MaxStocks")

        model.setParam(GRB.Param.IterationLimit, max_iterations)

        if(initial_solution):
            model.setAttr("Start", w, initial_solution['w'])
            model.setAttr("Start", z, initial_solution['z'])

        model.optimize()

        result = {
            'root_mean_squared_error': model.ObjVal,
            'portfolio': {i: w[i].x for i in I if z[i].x}
        }
        
        return result
    
    def mult_start_optimization(self, 
                                portfolio_size=DEFAULT_PORTFOLIO_SIZE, 
                                num_starts=DEFAULT_MULTI_START_ITERATIONS, 
                                max_iterations=DEFAULT_MAX_ITERATIONS):
        """
        Perform multi-start optimization to find the best portfolio across train and test data
        
        Parameters:
        - portfolio_size: Number of stocks to select
        - num_starts: Number of different starting points
        - max_iterations: Maximum iterations per run
        
        Returns:
        - Best portfolio dictionary with performance metrics
        """
        if self.train_stock_data is None:
            print('Error')
            return
        
        best_result = {
            'error': float('inf'),
            'portfolio': None,
            'performance': None
        }

        best_result_is = {
            'error': float('inf'),
            'portfolio': None,
            'performance': None
        }

        results = []
        results_is = []
        
        for start in range(num_starts):
            try:
                # porfolio
                current_portfolio = self.create_portfolio(
                    portfolio_size=portfolio_size, 
                    max_iterations=max_iterations
                )

                performance = self.compare_train_test_performance(current_portfolio['portfolio'])
                
                test_rmse = performance['test_performance']['root_mean_squared_error']

                new_result = {
                    'error': test_rmse,
                    'portfolio': current_portfolio['portfolio'],
                    'performance': performance
                }

                results.append(new_result)
                                
                if test_rmse < best_result['error']:
                    best_result = new_result
                
                # portfolio with initial solution
                w_initial, z_initial = self.generate_initial_solution(portfolio_size)
                initial_solution = {'w': w_initial, 'z': z_initial}
                current_portfolio_is = self.create_portfolio(
                    portfolio_size=portfolio_size, 
                    max_iterations=max_iterations,
                    initial_solution=initial_solution
                )

                performance_is = self.compare_train_test_performance(current_portfolio['portfolio'])
                
                test_rmse_is = performance['test_performance']['root_mean_squared_error']

                new_result_is = {
                    'error': test_rmse_is,
                    'portfolio': current_portfolio_is['portfolio'],
                    'performance': performance_is
                }

                results_is.append(new_result_is)
                                
                if test_rmse_is < best_result_is['error']:
                    best_result_is = new_result_is
                
            
            except gp.GurobiError as e:
                print(f"Optimization error in iteration {start}: {e}")
                continue
        
        self.best_result = best_result
        self.results = sorted(results, key=lambda x: x['error'])

        self.best_result_is = best_result_is
        self.results_is = sorted(results_is, key=lambda x: x['error'])
        
        return best_result, results
    
    def calculate_tracking_error(self, portfolio_weights, is_train=True):
        """
        Calculate tracking error metrics for the portfolio
        
        Parameters:
        - portfolio_weights: Dictionary of stock weights
        - is_train: Whether to use training or testing data
        
        Returns:
        - Dictionary of tracking error metrics
        """
        if is_train:
            stock_data = self.train_stock_data
            index_data = self.train_index_data
        else:
            stock_data = self.test_stock_data
            index_data = self.test_index_data
        
        selected_stocks = list(portfolio_weights.keys())
        selected_stock_returns = stock_data[selected_stocks]
        
        weighted_portfolio_returns = (selected_stock_returns * list(portfolio_weights.values())).sum(axis=1)
        
        tracking_error_metrics = {
            'tracking_error': np.std(weighted_portfolio_returns - index_data),
            'root_mean_squared_error': np.sqrt(np.mean((weighted_portfolio_returns - index_data)**2)),
            'correlation': np.corrcoef(weighted_portfolio_returns, index_data)[0, 1],
        }
        
        return tracking_error_metrics
    
    def compare_train_test_performance(self, portfolio_weights):
        """
        Compare portfolio performance on training and testing data
        
        Parameters:
        - portfolio_weights: Dictionary of stock weights
        
        Returns:
        - Dictionary with train and test performance metrics
        """
        train_metrics = self.calculate_tracking_error(portfolio_weights, is_train=True)
        test_metrics = self.calculate_tracking_error(portfolio_weights, is_train=False)
        
        return {
            'train_performance': train_metrics,
            'test_performance': test_metrics
        }
    
    def generate_initial_solution(self, portfolio_size=DEFAULT_PORTFOLIO_SIZE):
        """
        Generate initial solution for the portfolio optimization problem.
        
        Parameters:
        - stock_data: DataFrame with stock return data
        - index_data: Array with index return data
        - portfolio_size: Number of stocks to select (k)
        
        Returns:
        - Tuple (w_initial, z_initial) where:
        - w_initial: Dictionary with initial weights for stocks
        - z_initial: Dictionary with initial binary selection for stocks
        """

        if self.train_stock_data is None:
            print('Error')
            return

        stock_data = self.train_stock_data
        index_data = self.train_index_data

        I = np.array(stock_data.columns)
        T = range(len(stock_data))
        r = {(t, i): stock_data.iloc[t][i] for t in T for i in I}
        R = np.array(index_data)
        k = portfolio_size

        relaxed_model = gp.Model("RelaxedPortfolio")
        w_relaxed = relaxed_model.addVars(I, lb=0, ub=1, name="w")

        relaxed_model.setObjective(
            (1 / len(T)) * gp.quicksum(
                (gp.quicksum(w_relaxed[i] * r[t, i] for i in I) - R[t]) ** 2 for t in T
            ),
            GRB.MINIMIZE
        )

        relaxed_model.addConstr(gp.quicksum(w_relaxed[i] for i in I) == 1, "WeightSum")
        relaxed_model.optimize()

        if relaxed_model.Status != GRB.OPTIMAL:
            print("Relaxed model did not converge.")
            return None, None

        relaxed_weights = {i: w_relaxed[i].x for i in I}

        top_k_stocks = sorted(relaxed_weights, key=relaxed_weights.get, reverse=True)[:k]

        w_initial = {i: relaxed_weights[i] if i in top_k_stocks else 0 for i in I}
        total_weight_top_k = sum(w_initial[i] for i in top_k_stocks)
        w_initial = {i: (w_initial[i] / total_weight_top_k) if i in top_k_stocks else 0 for i in I}
        z_initial = {i: 1 if i in top_k_stocks else 0 for i in I}

        return w_initial, z_initial
