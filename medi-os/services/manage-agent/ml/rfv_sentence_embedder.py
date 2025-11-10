"""
RFV Sentence Embedder Module

Converts RFV text labels to dense semantic embeddings using sentence transformers.
This captures semantic meaning beyond keyword matching.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

from ml.rfv_mapper import RFVTextToCodeMapper


class RFVSentenceEmbedder(BaseEstimator, TransformerMixin):
    """
    Converts RFV codes to sentence embeddings using pre-trained models.
    
    Uses sentence-transformers to generate dense semantic embeddings (e.g., 384 dims)
    from RFV text labels, capturing semantic relationships better than keyword matching.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        rfv_columns: Optional[List[str]] = None,
        embedding_dim: Optional[int] = None,
        use_pca: bool = False,
        pca_components: int = 20,
        cache_embeddings: bool = True
    ):
        """
        Initialize RFV sentence embedder.
        
        Args:
            model_name: Sentence transformer model name
                       Options: "all-MiniLM-L6-v2" (384 dims, fast)
                               "all-mpnet-base-v2" (768 dims, better quality)
                               Medical domain models if available
            rfv_columns: List of RFV column names (default: ['rfv1', 'rfv2'])
            embedding_dim: Expected embedding dimension (auto-detected if None)
            use_pca: If True, reduce embeddings via PCA to pca_components dimensions
            pca_components: Number of PCA components if use_pca=True
            cache_embeddings: If True, cache embeddings for faster processing
        """
        self.model_name = model_name
        self.rfv_columns = rfv_columns or ['rfv1', 'rfv2']
        self.embedding_dim = embedding_dim
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.cache_embeddings = cache_embeddings
        
        # Will be initialized in fit()
        self.model = None
        self.rfv_mapper = None
        self.code_to_embedding: Dict[float, np.ndarray] = {}
        self.pca = None
        self.embedding_cache_path = None
    
    def _load_model(self):
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Auto-detect embedding dimension
            if self.embedding_dim is None:
                # Test with a sample sentence
                test_embedding = self.model.encode(["test"], convert_to_numpy=True)
                self.embedding_dim = test_embedding.shape[1]
                print(f"  Auto-detected embedding dimension: {self.embedding_dim}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence transformer model: {e}")
    
    def _load_rfv_mappings(self):
        """Load RFV code-to-text mappings."""
        try:
            self.rfv_mapper = RFVTextToCodeMapper.load()
            print(f"  Loaded RFV mappings: {len(self.rfv_mapper._code_to_text_map)} fields")
        except FileNotFoundError:
            raise FileNotFoundError(
                "RFV mappings not found. Run parser to generate rfv_code_mappings.json"
            )
    
    def _generate_embeddings(self, rfv_column: str) -> Dict[float, np.ndarray]:
        """
        Generate embeddings for all RFV codes in a column.
        
        Args:
            rfv_column: RFV column name (e.g., 'rfv1')
            
        Returns:
            Dictionary mapping RFV code to embedding vector
        """
        code_to_text = self.rfv_mapper._code_to_text_map.get(rfv_column, {})
        
        if not code_to_text:
            print(f"  Warning: No mappings found for {rfv_column}")
            return {}
        
        # Prepare texts for encoding
        codes = []
        texts = []
        for code_str, text in code_to_text.items():
            try:
                code = float(code_str)
                codes.append(code)
                texts.append(text)
            except (ValueError, TypeError):
                continue
        
        if not texts:
            return {}
        
        print(f"  Generating embeddings for {len(texts)} {rfv_column} codes...")
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings_list = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings_list.append(batch_embeddings)
        
        # Combine all embeddings
        all_embeddings = np.vstack(embeddings_list)
        
        # Create code-to-embedding mapping
        code_to_embedding = {}
        for code, embedding in zip(codes, all_embeddings):
            code_to_embedding[code] = embedding
        
        # Handle missing/zero codes (use zero vector)
        zero_embedding = np.zeros(self.embedding_dim)
        code_to_embedding[0.0] = zero_embedding
        
        print(f"  Generated {len(code_to_embedding)} embeddings for {rfv_column}")
        
        return code_to_embedding
    
    def _apply_pca(self, embeddings_dict: Dict[float, np.ndarray]) -> Dict[float, np.ndarray]:
        """
        Apply PCA to reduce embedding dimensions.
        
        Args:
            embeddings_dict: Dictionary mapping code to embedding
            
        Returns:
            Dictionary with reduced-dimension embeddings
        """
        if not embeddings_dict:
            return {}
        
        from sklearn.decomposition import PCA
        
        # Stack all embeddings
        embeddings_array = np.array(list(embeddings_dict.values()))
        
        # Fit PCA
        self.pca = PCA(n_components=self.pca_components)
        reduced_embeddings = self.pca.fit_transform(embeddings_array)
        
        # Create new mapping
        reduced_dict = {}
        for i, code in enumerate(embeddings_dict.keys()):
            reduced_dict[code] = reduced_embeddings[i]
        
        print(f"  Applied PCA: {self.embedding_dim} → {self.pca_components} dimensions")
        
        return reduced_dict
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the embedder by generating embeddings for all RFV codes."""
        print(f"\n[RFV Sentence Embedder] Initializing...")
        
        # Load model and mappings
        self._load_model()
        self._load_rfv_mappings()
        
        # Set up cache path
        if self.cache_embeddings:
            cache_dir = project_root / "services" / "manage-agent" / "outputs"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.embedding_cache_path = cache_dir / f"rfv_embeddings_{self.model_name.replace('/', '_')}.pkl"
        
        # Try to load cached embeddings
        if self.cache_embeddings and self.embedding_cache_path.exists():
            print(f"  Loading cached embeddings from: {self.embedding_cache_path}")
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.code_to_embedding = cached_data.get('code_to_embedding', {})
                    self.pca = cached_data.get('pca', None)
                    # Update embedding_dim from cache if available (accounts for PCA reduction)
                    cached_dim = cached_data.get('embedding_dim', None)
                    if cached_dim is not None:
                        self.embedding_dim = cached_dim
                    elif self.code_to_embedding:
                        # Infer dimension from first embedding if not in cache
                        first_embedding = next(iter(self.code_to_embedding.values()))
                        self.embedding_dim = len(first_embedding)
                    if self.code_to_embedding:
                        print(f"  Loaded {len(self.code_to_embedding)} cached embeddings")
                        if self.pca:
                            print(f"  Using PCA-reduced dimension: {self.embedding_dim}")
                        return self
            except Exception as e:
                print(f"  Warning: Failed to load cache: {e}. Regenerating...")
        
        # Generate embeddings for each RFV column
        all_code_to_embedding = {}
        
        for rfv_col in self.rfv_columns:
            col_embeddings = self._generate_embeddings(rfv_col)
            all_code_to_embedding.update(col_embeddings)
        
        self.code_to_embedding = all_code_to_embedding
        
        # Apply PCA if requested
        if self.use_pca and self.code_to_embedding:
            self.code_to_embedding = self._apply_pca(self.code_to_embedding)
            self.embedding_dim = self.pca_components
        
        # Cache embeddings
        if self.cache_embeddings and self.code_to_embedding:
            cache_data = {
                'code_to_embedding': self.code_to_embedding,
                'pca': self.pca,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim
            }
            with open(self.embedding_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"  Cached embeddings to: {self.embedding_cache_path}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform RFV codes to embedding vectors.
        
        Args:
            X: DataFrame with RFV columns
            
        Returns:
            DataFrame with RFV columns replaced by embedding features
        """
        X = X.copy()
        
        if not self.code_to_embedding:
            raise ValueError("RFVSentenceEmbedder not fitted. Call fit() first.")
        
        # Process each RFV column
        for rfv_col in self.rfv_columns:
            if rfv_col not in X.columns:
                continue
            
            # Get embeddings for all codes in this column
            embeddings_list = []
            for code in X[rfv_col].values:
                code_float = float(code) if pd.notna(code) else 0.0
                embedding = self.code_to_embedding.get(
                    code_float,
                    np.zeros(self.embedding_dim)  # Default to zero vector
                )
                embeddings_list.append(embedding)
            
            # Create DataFrame with embedding columns
            embedding_df = pd.DataFrame(
                np.array(embeddings_list),
                columns=[f"{rfv_col}_emb_{i}" for i in range(self.embedding_dim)],
                index=X.index
            )
            
            # Drop original RFV column and add embeddings
            X = X.drop(columns=[rfv_col])
            X = pd.concat([X, embedding_df], axis=1)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

