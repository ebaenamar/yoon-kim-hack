"""
Tropical Attention Tasks - Dynamic Programming Combinatorial Problems
Based on "Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms"

These tasks require max-plus semiring operations and sharp polyhedral reasoning.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class TropicalTaskConfig:
    """Configuration for Tropical tasks"""
    max_nodes: int = 20  # Maximum nodes in graph
    max_value: int = 100  # Maximum edge weight
    seq_length: int = 256
    seed: int = 42


class LongestPathTask:
    """
    Longest Path in DAG using dynamic programming
    Tests if model can perform max-plus operations
    """
    
    def __init__(self, config: TropicalTaskConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
    def generate_dag(self, num_nodes: int) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """
        Generate a random DAG with weights
        
        Returns:
            adjacency matrix and list of (from, to, weight) edges
        """
        adj = np.zeros((num_nodes, num_nodes), dtype=np.int32)
        edges = []
        
        # Generate edges only from lower to higher indices (ensures DAG)
        for i in range(num_nodes - 1):
            # Connect to 1-3 random nodes ahead
            num_edges = self.rng.randint(1, min(4, num_nodes - i))
            targets = self.rng.choice(
                range(i + 1, num_nodes), 
                size=num_edges, 
                replace=False
            )
            
            for j in targets:
                weight = self.rng.randint(1, self.config.max_value)
                adj[i, j] = weight
                edges.append((i, j, weight))
        
        return adj, edges
    
    def compute_longest_path(self, adj: np.ndarray) -> np.ndarray:
        """
        Compute longest path from node 0 to all other nodes using DP
        
        Returns:
            Array of longest path distances
        """
        num_nodes = len(adj)
        dp = np.full(num_nodes, -np.inf)
        dp[0] = 0
        
        # DP in topological order (our indices are already sorted)
        for i in range(num_nodes):
            if dp[i] == -np.inf:
                continue
            for j in range(i + 1, num_nodes):
                if adj[i, j] > 0:
                    dp[j] = max(dp[j], dp[i] + adj[i, j])
        
        return dp
    
    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a longest path problem as a sequence
        
        Format: [num_nodes] [edge1_from edge1_to edge1_weight] ... [QUERY] [target_distances]
        """
        num_nodes = self.rng.randint(5, self.config.max_nodes)
        adj, edges = self.generate_dag(num_nodes)
        distances = self.compute_longest_path(adj)
        
        # Encode as sequence
        sequence = [num_nodes]
        
        # Add edges
        for from_node, to_node, weight in edges:
            sequence.extend([from_node, to_node, weight])
        
        # Add query marker (special token)
        query_marker = self.config.max_value + 100
        sequence.append(query_marker)
        
        # Targets: longest distances to each node
        targets = sequence[1:] + [0]  # Shift for next token prediction
        
        # For this task, we primarily care about final predictions
        # But we can also check intermediate DP steps
        
        # Mask: evaluate on distance predictions after query marker
        mask = torch.zeros(len(sequence), dtype=torch.bool)
        # We'll evaluate the entire output in a special way
        
        return (
            torch.tensor(sequence, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            mask,
            {'distances': distances, 'num_nodes': num_nodes}
        )


class KnapsackTask:
    """
    0-1 Knapsack problem
    Tests max operation and dynamic programming over items and capacity
    """
    
    def __init__(self, config: TropicalTaskConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
    
    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Generate a knapsack problem
        
        Format: [capacity] [n_items] [w1 v1] [w2 v2] ... [QUERY] [max_value]
        """
        capacity = self.rng.randint(10, 50)
        n_items = self.rng.randint(5, 15)
        
        weights = self.rng.randint(1, capacity // 2, size=n_items)
        values = self.rng.randint(1, 100, size=n_items)
        
        # Solve knapsack
        max_value = self.solve_knapsack(capacity, weights, values)
        
        # Encode as sequence
        sequence = [capacity, n_items]
        for w, v in zip(weights, values):
            sequence.extend([w, v])
        
        query_marker = self.config.max_value + 100
        sequence.append(query_marker)
        sequence.append(max_value)
        
        targets = sequence[1:]
        
        # Mask: only evaluate on the final max_value prediction
        mask = torch.zeros(len(targets), dtype=torch.bool)
        mask[-1] = True

        return (
            torch.tensor(sequence[:-1], dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            mask
        )
    
    def solve_knapsack(self, capacity: int, weights: np.ndarray, values: np.ndarray) -> int:
        """Solve 0-1 knapsack using DP"""
        n = len(weights)
        dp = np.zeros(capacity + 1, dtype=np.int32)
        
        for i in range(n):
            for w in range(capacity, weights[i] - 1, -1):
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
        
        return dp[capacity]


class EditDistanceTask:
    """
    Edit Distance (Levenshtein Distance)
    Tests min operation and DP over two sequences
    """
    
    def __init__(self, config: TropicalTaskConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.alphabet_size = 26
    
    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Generate an edit distance problem
        
        Format: [len1] [seq1] [len2] [seq2] [QUERY] [distance]
        """
        len1 = self.rng.randint(5, 15)
        len2 = self.rng.randint(5, 15)
        
        seq1 = self.rng.randint(0, self.alphabet_size, size=len1)
        seq2 = self.rng.randint(0, self.alphabet_size, size=len2)
        
        # Compute edit distance
        distance = self.compute_edit_distance(seq1, seq2)
        
        # Encode as sequence
        sequence = [len1] + seq1.tolist() + [len2] + seq2.tolist()
        query_marker = self.config.max_value + 100
        sequence.extend([query_marker, distance])
        
        targets = sequence[1:]
        
        return (
            torch.tensor(sequence[:-1], dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            {'distance': distance, 'len1': len1, 'len2': len2}
        )
    
    def compute_edit_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> int:
        """Compute edit distance using DP"""
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=np.int32)
        
        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1]
                else:
                    dp[i, j] = 1 + min(
                        dp[i - 1, j],     # delete
                        dp[i, j - 1],     # insert
                        dp[i - 1, j - 1]  # replace
                    )
        
        return dp[m, n]


def evaluate_tropical_task(model, task, num_samples: int = 100, device='cuda') -> dict:
    """
    Evaluate model on tropical/DP tasks
    
    Args:
        model: Model with forward(input_ids) -> logits method
        task: Task instance (LongestPathTask, KnapsackTask, etc.)
        num_samples: Number of samples to evaluate
        device: Device to run on
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for _ in range(num_samples):
            if isinstance(task, LongestPathTask):
                inputs, targets, mask, info = task.generate_sample()
                # Special evaluation for longest path
                # Would need to extract final predictions and compare with info['distances']
                total_samples += 1
            else:
                inputs, targets, info = task.generate_sample()
                inputs = inputs.unsqueeze(0).to(device)
                targets = targets.unsqueeze(0).to(device)
                
                # Get model predictions
                logits = model(inputs)
                predictions = logits.argmax(dim=-1)
                
                # Check if final prediction matches
                final_pred = predictions[0, -1].item()
                if isinstance(task, KnapsackTask):
                    final_target = info['max_value']
                elif isinstance(task, EditDistanceTask):
                    final_target = info['distance']
                
                if final_pred == final_target:
                    total_correct += 1
                total_samples += 1
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'total_correct': total_correct,
        'total_samples': total_samples,
        'task': task.__class__.__name__
    }


def test_tropical_tasks():
    """Test tropical task generation"""
    config = TropicalTaskConfig(max_nodes=10)
    
    print("=== Longest Path Task ===")
    lp_task = LongestPathTask(config)
    inputs, targets, mask, info = lp_task.generate_sample()
    print(f"Input length: {len(inputs)}")
    print(f"Distances: {info['distances']}")
    print(f"Num nodes: {info['num_nodes']}")
    
    print("\n=== Knapsack Task ===")
    ks_task = KnapsackTask(config)
    inputs, targets, info = ks_task.generate_sample()
    print(f"Input length: {len(inputs)}")
    print(f"Max value: {info['max_value']}")
    print(f"Capacity: {info['capacity']}")
    
    print("\n=== Edit Distance Task ===")
    ed_task = EditDistanceTask(config)
    inputs, targets, info = ed_task.generate_sample()
    print(f"Input length: {len(inputs)}")
    print(f"Edit distance: {info['distance']}")


if __name__ == '__main__':
    test_tropical_tasks()
