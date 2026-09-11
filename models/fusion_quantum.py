"""Quantum Fusion Module using PennyLane simulation on default.qubit.

Per Section 7.5:
- 6 qubits: wires 0, 1, 2 for video; wires 3, 4, 5 for audio
- Shared compression: Linear(768, 3) per modality -> 6 angles
- Angle embedding into qubits using RY rotations
- 4 variational rotation layers with 72 trainable parameters total (4 layers * 6 qubits * 3 rotations)
- Explicit cross-modal entangling CNOT gates on wires [2, 3] and [5, 0]
- Measurement: PauliZ expectation values across all 6 qubits -> 6-vector
- Shared expansion: Linear(6, 512) + LayerNorm(512)
"""

import torch
import torch.nn as nn
import pennylane as qml


class QuantumFusion(nn.Module):
    def __init__(
        self,
        video_dim: int = 768,
        audio_dim: int = 768,
        fused_dim: int = 512,
        num_qubits: int = 6,
        num_layers: int = 4
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # 1. Classical compression infrastructure (identical to ClassicalMatchedFusion)
        self.video_compress = nn.Linear(video_dim, 3)
        self.audio_compress = nn.Linear(audio_dim, 3)

        # 2. Trainable quantum circuit parameters: [num_layers, num_qubits, 3] = 4 * 6 * 3 = 72 parameters
        self.quantum_weights = nn.Parameter(
            0.01 * torch.randn(num_layers, num_qubits, 3, dtype=torch.float32)
        )

        # 3. PennyLane Quantum Device & QNode
        self.dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            # Angle embedding
            for w in range(num_qubits):
                qml.RY(inputs[:, w], wires=w)

            # 4 Variational Layers
            for l in range(num_layers):
                for w in range(num_qubits):
                    qml.Rot(weights[l, w, 0], weights[l, w, 1], weights[l, w, 2], wires=w)

                # Explicit cross-modal CNOT wiring per spec
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[5, 0])

                # Intra-modality / ring entanglement
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[3, 4])
                qml.CNOT(wires=[4, 5])

            # Measure expectation values of PauliZ on all 6 qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.circuit = _circuit

        # 4. Classical expansion infrastructure (identical to ClassicalMatchedFusion)
        self.expand = nn.Linear(num_qubits, fused_dim)
        self.layernorm = nn.LayerNorm(fused_dim)

    def draw_circuit(self) -> str:
        """Returns the ASCII diagram of the quantum fusion circuit."""
        dev = qml.device("default.qubit", wires=self.num_qubits)
        @qml.qnode(dev, interface="torch")
        def _draw_qnode(inputs, weights):
            for w in range(self.num_qubits):
                qml.RY(inputs[w], wires=w)
            for l in range(self.num_layers):
                for w in range(self.num_qubits):
                    qml.Rot(weights[l, w, 0], weights[l, w, 1], weights[l, w, 2], wires=w)
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[5, 0])
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[3, 4])
                qml.CNOT(wires=[4, 5])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        dummy_in = torch.zeros(self.num_qubits)
        dummy_weights = torch.zeros(self.num_layers, self.num_qubits, 3)
        return str(qml.draw(_draw_qnode)(dummy_in, dummy_weights))

    def forward(self, video_feats: torch.Tensor, audio_feats: torch.Tensor) -> torch.Tensor:
        """Forward pass for quantum fusion.
        
        Args:
            video_feats: Tensor of shape [..., T, 768]
            audio_feats: Tensor of shape [..., T, 768]
        Returns:
            fused_feats: Tensor of shape [..., T, 512]
        """
        orig_shape = video_feats.shape[:-1]

        # Flatten leading dimensions to [B*T, 768]
        v_flat = video_feats.reshape(-1, video_feats.shape[-1])
        a_flat = audio_feats.reshape(-1, audio_feats.shape[-1])

        # 1. Classical compression: [B*T, 3] + [B*T, 3] -> [B*T, 6]
        v_comp = self.video_compress(v_flat)
        a_comp = self.audio_compress(a_flat)
        state = torch.cat([v_comp, a_comp], dim=-1) # [B*T, 6]

        # 2. Quantum execution on default.qubit (CPU simulator)
        in_device = state.device
        state_cpu = state.to("cpu")
        w_cpu = self.quantum_weights.to("cpu")

        q_out = self.circuit(state_cpu, w_cpu)
        # Stack expectation values into [B*T, 6] tensor (cast to float32 and move back to original device)
        q_meas = torch.stack(q_out, dim=-1).float().to(in_device)

        # 3. Classical expansion
        expanded = self.expand(q_meas) # [B*T, 512]
        normed = self.layernorm(expanded)

        # Reshape back to original leading dimensions
        return normed.reshape(*orig_shape, -1)
