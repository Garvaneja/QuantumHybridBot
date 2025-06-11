import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.preprocessing import RobustScaler

# Configure loggers
logger = logging.getLogger("core.quantum_hybrid")
logger.setLevel(logging.INFO)
# Suppress verbose websocket output
logging.getLogger("websockets.protocol").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

class QuantumHybrid(nn.Module):
    """
    Simplified hybrid "quantum"–classical model using a small feed‑forward network
    to simulate a quantum circuit, avoiding Qiskit entirely.
    """
    def __init__(self, num_qubits: int = 4):
        super().__init__()
        self._input_dim = num_qubits
        self.scaler = RobustScaler()

        # "Quantum" simulator (classical NN)
        self.quantum_simulator = nn.Sequential(
            nn.Linear(self._input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Softmax(dim=-1)
        )

        # Classical head: 2→1
        self.classical_fc = nn.Linear(2, 1)

        # Sentiment gating
        self.sentiment_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.trained = False
        self.loss_history = []

    def _dict_to_vec(self, price_data: dict) -> np.ndarray:
        vec = np.array([
            price_data.get("momentum", 0.0),
            np.log1p(abs(price_data.get("volatility", 0.0))),
            (price_data.get("rsi", 50.0) - 50) / 50,
            np.clip(price_data.get("obv", 0.0) / 1e6, -1, 1)
        ], dtype=float)

        if not hasattr(self.scaler, 'scale_'):
            self.scaler.fit(vec.reshape(1, -1))
        norm = self.scaler.transform(vec.reshape(1, -1)).flatten()
        return self._ensure_length(norm)

    def _array_to_vec(self, arr: np.ndarray) -> np.ndarray:
        vec = np.asarray(arr, dtype=float).flatten()
        return self._ensure_length(vec)

    def _ensure_length(self, vec: np.ndarray) -> np.ndarray:
        if vec.size > self._input_dim:
            return vec[:self._input_dim]
        if vec.size < self._input_dim:
            pad = np.zeros(self._input_dim - vec.size, dtype=float)
            return np.concatenate([vec, pad])
        return vec

    def quantum_forward(self, x) -> torch.Tensor:
        try:
            vec = self._dict_to_vec(x) if isinstance(x, dict) else self._array_to_vec(x)
            features = torch.tensor(vec, dtype=torch.float32)
            probs = self.quantum_simulator(features)
            signal = torch.sigmoid(self.classical_fc(probs))
            return signal
        except Exception as e:
            logger.error(f"Quantum forward error: {e}")
            return torch.tensor([0.5], dtype=torch.float32)

    async def train(self, X_train, y_train, epochs: int = 50, lr: float = 0.01, verbose: bool = False):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        X = list(X_train) if not isinstance(X_train, np.ndarray) else [X_train[i] for i in range(len(X_train))]
        y = torch.tensor(y_train, dtype=torch.float32)

        if verbose:
            logger.info(f"Training with {len(X)} samples for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            preds = torch.stack([self.quantum_forward(x).squeeze() for x in X])
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            if verbose and (epoch == 1 or epoch % 10 == 0):
                logger.info(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

        self.trained = True
        if verbose:
            logger.info("Training completed successfully.")

    def predict(self, price_data: dict, sentiment_score: float = 0.0) -> float:
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")
        with torch.no_grad():
            q_sig = self.quantum_forward(price_data).squeeze()
            gate = torch.sigmoid(self.sentiment_weight * sentiment_score * 2)
            hybrid = q_sig * (1 + gate) / 2
            return float(torch.clamp(hybrid, 0, 1).item())


class QuantumHybridQiskit(nn.Module):
    """
    Quantum-classical hybrid using Qiskit SamplerQNN, with a robust fallback to the classical simulator.
    """
    def __init__(self, num_qubits: int = 4):
        super().__init__()
        self._input_dim = num_qubits
        self.scaler = RobustScaler()

        # Attempt Qiskit initialization
        try:
            from qiskit_machine_learning.neural_networks import SamplerQNN
            from qiskit.circuit.library import ZZFeatureMap
            from qiskit.circuit import ParameterVector, QuantumCircuit
            self.qnn, self._input_params, self._weight_params = self._build_qnn()
            self.use_qnn = True
            logger.info("Qiskit QNN initialized successfully.")
        except Exception as e:
            logger.warning(f"Qiskit init failed: {e}. Using classical fallback.")
            self.use_qnn = False
            self.quantum_simulator = nn.Sequential(
                nn.Linear(self._input_dim, 8),
                nn.Tanh(),
                nn.Linear(8, 2),
                nn.Softmax(dim=-1)
            )

        # Weight parameter setup
        w_size = len(self._weight_params) if self.use_qnn else self._input_dim * 2
        self.q_weights = nn.Parameter(torch.randn(w_size), requires_grad=True)
        self.classical_fc = nn.Linear(2, 1)
        self.sentiment_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.trained = False
        self.loss_history = []

    def _build_qnn(self):
        from qiskit_machine_learning.neural_networks import SamplerQNN
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit.circuit import ParameterVector, QuantumCircuit

        feature_map = ZZFeatureMap(feature_dimension=self._input_dim, reps=1, entanglement='linear')
        ansatz = QuantumCircuit(self._input_dim)
        params = ParameterVector('θ', length=3 * self._input_dim)
        # Simple ansatz
        for i in range(self._input_dim):
            ansatz.ry(params[i], i)
        for i in range(self._input_dim - 1):
            ansatz.cx(i, i+1)
            ansatz.rz(params[self._input_dim + i], i+1)
            ansatz.cx(i, i+1)
        for i in range(self._input_dim):
            ansatz.ry(params[2*self._input_dim + i], i)

        qc = QuantumCircuit(self._input_dim)
        qc.append(feature_map.to_instruction(), range(self._input_dim))
        qc.append(ansatz.to_instruction(), range(self._input_dim))

        def interpret(counts):
            total = max(1, sum(counts.values()))
            p1 = sum(v for k,v in counts.items() if k.endswith('1')) / total
            return [p1, 1-p1]

        qnn = SamplerQNN(circuit=qc,
                         input_params=feature_map.parameters,
                         weight_params=params,
                         interpret=interpret,
                         output_shape=2)
        return qnn, list(feature_map.parameters), list(params)

    def _dict_to_vec(self, price_data: dict) -> np.ndarray:
        vec = np.array([
            price_data.get("momentum", 0.0),
            np.log1p(abs(price_data.get("volatility", 0.0))),
            (price_data.get("rsi", 50.0) - 50) / 50,
            np.clip(price_data.get("obv", 0.0) / 1e6, -1, 1)
        ], dtype=float)
        if not hasattr(self.scaler, 'scale_'):
            self.scaler.fit(vec.reshape(1, -1))
        norm = self.scaler.transform(vec.reshape(1, -1)).flatten()
        return self._ensure_length(norm)

    def _array_to_vec(self, arr: np.ndarray) -> np.ndarray:
        return self._ensure_length(np.asarray(arr, dtype=float).flatten())

    def _ensure_length(self, vec: np.ndarray) -> np.ndarray:
        if vec.size > self._input_dim:
            return vec[:self._input_dim]
        if vec.size < self._input_dim:
            pad = np.zeros(self._input_dim - vec.size, dtype=float)
            return np.concatenate([vec, pad])
        return vec

    def _classical_forward(self, vec):
        x = torch.tensor(vec, dtype=torch.float32)
        probs = self.quantum_simulator(x) if not self.use_qnn else torch.softmax(x[:2], dim=0)
        return torch.sigmoid(self.classical_fc(probs))

    def quantum_forward(self, x) -> torch.Tensor:
        try:
            vec = self._dict_to_vec(x) if isinstance(x, dict) else self._array_to_vec(x)
            if self.use_qnn:
                inp = vec.astype(np.float64)
                wt = self.q_weights.detach().cpu().numpy().astype(np.float64)
                raw = self.qnn.forward(inp, wt)
                arr = np.asarray(raw, dtype=float).flatten()
                if arr.size < 2:
                    arr = np.array([arr[0], 1 - arr[0]])
                probs = torch.softmax(torch.tensor(arr[:2], dtype=torch.float32), dim=0)
                return torch.sigmoid(self.classical_fc(probs))
            else:
                return self._classical_forward(vec)
        except Exception as e:
            logger.error(f"Forward error: {e}")
            return torch.tensor([0.5], dtype=torch.float32)

    async def train(self, X_train, y_train, epochs: int = 50, lr: float = 0.01, verbose: bool = False):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        X = list(X_train) if not isinstance(X_train, np.ndarray) else [X_train[i] for i in range(len(X_train))]
        y = torch.tensor(y_train, dtype=torch.float32)

        if verbose:
            logger.info(f"Training Qiskit hybrid with {len(X)} samples for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            preds = torch.stack([self.quantum_forward(x).squeeze() for x in X])
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            if verbose and (epoch == 1 or epoch % 10 == 0):
                logger.info(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

        self.trained = True
        if verbose:
            logger.info("Qiskit hybrid training completed successfully.")

    def predict(self, price_data: dict, sentiment_score: float = 0.0) -> float:
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")
        with torch.no_grad():
            q_sig = self.quantum_forward(price_data).squeeze()
            gate = torch.sigmoid(self.sentiment_weight * sentiment_score * 2)
            hybrid = q_sig * (1 + gate) / 2
            return float(torch.clamp(hybrid, 0, 1).item())
