from typing import Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class RMSNorm(nn.Module):
    """RMSNorm.

    Implements root mean squared normalization (Zhang et al., 2019).

    Example
    -------
    >>> module = RMSNorm()
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(self) -> None:
        """Initialize the module."""

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True))

        return x / rms


class Attention(nn.Module):
    """Attention.

    Implements multi-head self-attention (Vaswani et al., 2017).

    Example
    -------
    >>> module = Attention(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x, mask=None)
    """

    def __init__(self, embedding_dimension: int, number_of_heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        number_of_heads : int
            The number of attention heads.
        """

        super().__init__()

        self.number_of_heads = number_of_heads

        self.linears = nn.ModuleList([
            nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension * 3,
                bias=False,           
            ),
            nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension,
                bias=False,
            )
        ])
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        mask : torch.Tensor
            The attention mask (e.g. a causal mask).

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = rearrange(self.linears[0](x), 'b t (n h e) -> n b h t e', n=3,
                            h=self.number_of_heads)
        x = F.scaled_dot_product_attention(*x, attn_mask=mask)
        x = self.linears[1](rearrange(x, 'b h t e -> b t (h e)'))

        return x
    

class CoAttention(nn.Module):
    """Co-attention.

    Implements multi-head co-attention (Lu et al., 2019).

    Example
    -------
    >>> module = CoAttention(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> y = torch.randn((1, 20, 256))
    >>> x = module(x, y, mask=None)  # Shape: (1, 10, 256).
    """

    def __init__(self, embedding_dimension: int, number_of_heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        number_of_heads : int
            The number of attention heads.
        """

        super().__init__()

        self.number_of_heads = number_of_heads

        self.linears = nn.ModuleList([
            nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension,
                bias=False
            ) for _ in range(4)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The target input tensor.
        y : torch.Tensor
            The source input tensor.
        mask : torch.Tensor
            The attention mask.
        
        Returns
        -------
        x : torch.Tensor
            The target output tensor.
        """

        h = self.number_of_heads
        q = rearrange(self.linears[0](x), 'b t (h e) -> b h t e', h=h)
        k = rearrange(self.linears[1](y), 'b s (h e) -> b h s e', h=h)
        v = rearrange(self.linears[2](y), 'b s (h e) -> b h s e', h=h)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = self.linears[3](rearrange(x, 'b h t e -> b t (h e)'))

        return x


class MLP(nn.Module):
    """Implements an MLP (Vaswani et al., 2019).

    Example
    -------
    >>> module = MLP(embedding_dimension=256)
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(self, embedding_dimension: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                in_features=embedding_dimension,
                out_features=embedding_dimension * 4,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=embedding_dimension * 4,
                out_features=embedding_dimension,
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """ 

        return self.layers(x)


class TransformerBlock(nn.Module):
    """Implements a transformer block (Devlin et al., 2019).

    Example
    -------
    >>> module = TransformerBlock(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x, mask=None)
    """

    def __init__(self, embedding_dimension: int, number_of_heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        number_of_heads : int
            The number of attention heads.
        """

        super().__init__()

        self.attention = Attention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )

        self.mlp = MLP(embedding_dimension=embedding_dimension)
        self.rms_norms = nn.ModuleList([RMSNorm(), RMSNorm()])
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        mask : torch.Tensor
            The attention mask (e.g. a causal mask).
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = x + self.attention(self.rms_norms[0](x), mask=mask)
        x = x + self.mlp(self.rms_norms[1](x))

        return x


class CoTransformerBlock(nn.Module):
    """Co-transoformer block.

    Implements a co-transformer block (Lu et al., 2019).

    Example
    -------
    >>> module = CoTransformerBlock(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> y = torch.randn((1, 20, 256))
    >>> x = module(x, y, mask=None)  # Shape: (1, 10, 256).
    """

    def __init__(self, embedding_dimension: int, number_of_heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        number_of_heads : int
            The number of attention heads.
        """

        super().__init__()

        self.co_attention = CoAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )

        self.mlp = MLP(embedding_dimension=embedding_dimension)
        self.rms_norms = nn.ModuleList([RMSNorm(), RMSNorm()])
    
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The target input tensor.
        y : torch.Tensor
            The source in put tensor.
        mask : torch.Tensor
            The attention mask (e.g. a causal mask).
        
        Returns
        -------
        x : torch.Tensor
            The target output tensor.
        """

        x = x + self.co_attention(x=self.rms_norms[0](x), y=y, mask=mask)
        x = x + self.mlp(self.rms_norms[1](x))

        return x


class ViLBlock(nn.Module):
    """ViL block.

    Implements a ViL block. This consists of two co-transformer blocks for
    visual and linguistic representations. The queries of one block are used as 
    keys and values for the other and visa-versa.

    Example
    -------
    >>> module = ViLBERTBlock(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ... )
    >>> x1 = torch.randn((1, 10, 256))
    >>> x2 = torch.randn((1, 10, 256))
    >>> x1, x2 = module(x1, x2, mask=None)
    """

    def __init__(self, embedding_dimension: int, number_of_heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        number_of_heads : int
            The number of attention heads.
        """

        super().__init__()

        self.co_transformer_blocks = nn.ModuleList([
            CoTransformerBlock(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads
            ) for _ in range(2)
        ])

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads
            ) for _ in range(2)
        ])
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward the module.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor.
        x2 : torch.Tensor
            The second input tensor.
        mask : torch.Tensor
            The attention mask (e.g. a causal mask).
        
        Returns
        -------
        x1 : torch.Tensor
            The first output tensor.
        x2 : torch.Tensor
            The second output tensor.
        """

        x1, x2 = ( 
            self.co_transformer_blocks[0](x=x1, y=x2, mask=mask),
            self.co_transformer_blocks[1](x=x2, y=x1, mask=mask),
        )

        x1 = self.transformer_blocks[0](x1, mask=mask)
        x2 = self.transformer_blocks[1](x2, mask=mask)

        return x1, x2
    

@dataclass(frozen=True)
class ViLBERTConfiguration:
    embedding_dimension: int
    number_of_heads: int
    number_of_transformer_layers: int
    number_of_vil_layers: int


class ViLBERT(nn.Module):
    """ViLBERT.

    Implements ViLBERT (Lu et al., 2019). ViLBERT is a multi-modal encoder-only
    transformer architecture that utilizes co-attention to mix sparse visual and
    linguistic representations.

    Example
    -------
    >>> configuration = ViLBERTConfiguration(
    ...     embedding_dimension=256,
    ...     number_of_heads=16,
    ...     number_of_transformer_layers=8,
    ...     number_of_vil_layers=8,
    ... )
    >>> module = ViLBERT(configuration=configuration)
    >>> x1 = torch.randn((1, 10, 256))  # Visual.
    >>> x2 = torch.randn((1, 10, 256))  # Linguistic.
    >>> x1, x2 = module(x1, x2, mask=None)

    Notes
    -----
    - This implementation assumes tokens are already embedded. It does not 
      include embedding layers for either viusal or linguistic tokens.
    - This implementation does not use positional encoding, as this is typically
      added to token embeddings. However, ALiBi may be added in future. 
    """

    def __init__(self, configuration: ViLBERTConfiguration) -> None:
        """Initialize the module.

        Parameters
        ----------
        configuration : ViLBERTConfiguration
            The module configuration.
        """

        super().__init__()

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dimension=configuration.embedding_dimension,
                number_of_heads=configuration.number_of_heads,
            ) for _ in range(configuration.number_of_transformer_layers)
        ])

        self.vil_blocks = nn.ModuleList([
            ViLBlock(
                embedding_dimension=configuration.embedding_dimension,
                number_of_heads=configuration.number_of_heads,
            ) for _ in range(configuration.number_of_vil_layers)
        ])
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward the module.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor.
        x2 : torch.Tensor
            The second input tensor.
        mask : torch.Tensor
            The attention mask (e.g. a causal mask).
        
        Returns
        -------
        x1 : torch.Tensor
            The first output tensor.
        x2 : torch.Tensor
            The second output tensor.
        """

        for transformer_block in self.transformer_blocks:
            x2 = transformer_block(x2, mask=mask)
        
        for vil_block in self.vil_blocks:
            x1, x2 = vil_block(x1, x2, mask=mask)
        
        return x1, x2
