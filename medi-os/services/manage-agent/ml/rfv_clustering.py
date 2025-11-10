"""
RFV Clustering Module

Creates 10-15 meaningful clusters from 723 RFV codes using semantic grouping.
Uses text labels from .sps files to group similar medical complaints.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

from ml.rfv_mapper import RFVTextToCodeMapper


class RFVClusterer:
    """
    Clusters RFV codes into 10-15 meaningful medical categories.
    
    Uses semantic grouping based on text labels to create clusters like:
    - Cardiovascular/Chest
    - Respiratory
    - Gastrointestinal
    - Neurological
    - Musculoskeletal
    - etc.
    """
    
    # Medical domain clusters (10-15 clusters)
    CLUSTER_DEFINITIONS = {
        "Cardiovascular": {
            "keywords": ["chest", "heart", "cardiac", "palpitation", "pulse", "hypertension", "blood pressure"],
            "codes": []  # Will be populated
        },
        "Respiratory": {
            "keywords": ["breath", "cough", "wheez", "respir", "sputum", "shortness", "dyspnea", "asthma"],
            "codes": []
        },
        "Gastrointestinal": {
            "keywords": ["stomach", "abdominal", "nausea", "vomit", "diarrhea", "constipation", "bowel", "digestive"],
            "codes": []
        },
        "Neurological": {
            "keywords": ["headache", "head", "dizzy", "vertigo", "seizure", "convulsion", "neurolog", "fainting", "syncope"],
            "codes": []
        },
        "Musculoskeletal": {
            "keywords": ["back", "neck", "shoulder", "knee", "joint", "muscle", "fracture", "sprain", "pain", "ache", "discomfort"],
            "codes": []
        },
        "Skin": {
            "keywords": ["rash", "skin", "lesion", "wound", "burn", "infection", "dermatitis"],
            "codes": []
        },
        "Urinary_Genitourinary": {
            "keywords": ["urine", "urinary", "bladder", "kidney", "menstrual", "vaginal", "pelvic", "reproductive"],
            "codes": []
        },
        "Fever_Infection": {
            "keywords": ["fever", "infection", "chills", "sepsis", "viral", "bacterial"],
            "codes": []
        },
        "Trauma_Injury": {
            "keywords": ["injury", "trauma", "accident", "laceration", "cut", "puncture", "contusion", "abrasion"],
            "codes": []
        },
        "Mental_Health": {
            "keywords": ["depression", "anxiety", "psych", "suicide", "alcohol", "drug", "substance", "nervousness"],
            "codes": []
        },
        "Ear_Nose_Throat": {
            "keywords": ["ear", "nose", "throat", "soreness", "hearing", "vision", "eye", "sinus"],
            "codes": []
        },
        "General_Symptoms": {
            "keywords": ["weakness", "tiredness", "exhaustion", "ill feeling", "general"],
            "codes": []
        },
        "Other": {
            "keywords": [],  # Catch-all for unclassified
            "codes": []
        }
    }
    
    def __init__(self, mappings_path: Optional[str] = None):
        """
        Initialize RFV clusterer.
        
        Args:
            mappings_path: Path to rfv_code_mappings.json (auto-detect if None)
        """
        self.mapper = RFVTextToCodeMapper(mappings_path)
        self.code_to_cluster: Dict[float, str] = {}
        self.cluster_to_codes: Dict[str, List[float]] = {}
        self._build_clusters()
    
    def _build_clusters(self):
        """Build clusters by classifying RFV codes based on text labels."""
        # Load all RFV1 mappings
        rfv1_mappings = self.mapper._code_to_text_map.get('rfv1', {})
        
        # Reset cluster codes
        for cluster in self.CLUSTER_DEFINITIONS.values():
            cluster["codes"] = []
        
        # Classify each RFV code
        for code_str, label in rfv1_mappings.items():
            code = float(code_str)
            label_lower = label.lower()
            
            # Try to classify into a cluster
            classified = False
            for cluster_name, cluster_info in self.CLUSTER_DEFINITIONS.items():
                if cluster_name == "Other":
                    continue  # Skip "Other" until last
                
                keywords = cluster_info["keywords"]
                if any(keyword in label_lower for keyword in keywords):
                    cluster_info["codes"].append(code)
                    self.code_to_cluster[code] = cluster_name
                    classified = True
                    break
            
            # If not classified, add to "Other"
            if not classified:
                self.CLUSTER_DEFINITIONS["Other"]["codes"].append(code)
                self.code_to_cluster[code] = "Other"
        
        # Build reverse mapping
        for cluster_name, cluster_info in self.CLUSTER_DEFINITIONS.items():
            self.cluster_to_codes[cluster_name] = cluster_info["codes"]
        
        # Print summary
        print(f"RFV Clustering Summary:")
        for cluster_name, cluster_info in self.CLUSTER_DEFINITIONS.items():
            print(f"  {cluster_name}: {len(cluster_info['codes'])} codes")
    
    def get_cluster(self, code: float, default: str = "Other") -> str:
        """
        Get cluster name for an RFV code.
        
        Args:
            code: RFV code (float)
            default: Default cluster if code not found
            
        Returns:
            Cluster name (str)
        """
        return self.code_to_cluster.get(code, default)
    
    def fit_transform(self, df: pd.DataFrame, rfv_column: str = "rfv1") -> pd.DataFrame:
        """
        Transform RFV codes to cluster IDs.
        
        Args:
            df: DataFrame with RFV column
            rfv_column: Name of RFV column (default: "rfv1")
            
        Returns:
            DataFrame with new column {rfv_column}_cluster
        """
        df = df.copy()
        
        # Create cluster column
        cluster_col = f"{rfv_column}_cluster"
        df[cluster_col] = df[rfv_column].apply(
            lambda code: self.get_cluster(float(code) if pd.notna(code) else 0.0)
        )
        
        # Handle zeros/missing (0.0 codes)
        df.loc[df[rfv_column] == 0, cluster_col] = "Other"
        df.loc[df[rfv_column].isna(), cluster_col] = "Other"
        
        return df
    
    def get_cluster_counts(self, df: pd.DataFrame, rfv_column: str = "rfv1") -> pd.Series:
        """
        Get distribution of clusters in dataframe.
        
        Args:
            df: DataFrame with RFV column
            rfv_column: Name of RFV column
            
        Returns:
            Series with cluster counts
        """
        df_with_clusters = self.fit_transform(df, rfv_column)
        cluster_col = f"{rfv_column}_cluster"
        return df_with_clusters[cluster_col].value_counts()
    
    def save_cluster_mapping(self, output_path: str):
        """
        Save cluster mapping to JSON file.
        
        Args:
            output_path: Path to save JSON file
        """
        # Convert to JSON-serializable format
        json_mapping = {
            "code_to_cluster": {
                str(code): cluster for code, cluster in self.code_to_cluster.items()
            },
            "cluster_to_codes": {
                cluster: [str(code) for code in codes]
                for cluster, codes in self.cluster_to_codes.items()
            },
            "cluster_definitions": {
                name: {
                    "keywords": info["keywords"],
                    "code_count": len(info["codes"])
                }
                for name, info in self.CLUSTER_DEFINITIONS.items()
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"Saved cluster mapping to {output_path}")
    
    @classmethod
    def load_cluster_mapping(cls, mapping_path: str) -> 'RFVClusterer':
        """
        Load cluster mapping from JSON file.
        
        Args:
            mapping_path: Path to JSON file
            
        Returns:
            RFVClusterer instance
        """
        clusterer = cls.__new__(cls)  # Create instance without calling __init__
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            json_mapping = json.load(f)
        
        clusterer.code_to_cluster = {
            float(code): cluster for code, cluster in json_mapping["code_to_cluster"].items()
        }
        clusterer.cluster_to_codes = {
            cluster: [float(code) for code in codes]
            for cluster, codes in json_mapping["cluster_to_codes"].items()
        }
        
        return clusterer


def test_clustering():
    """Test RFV clustering on sample data."""
    print("=" * 80)
    print("TESTING RFV CLUSTERING")
    print("=" * 80)
    
    # Create clusterer
    print("\n[Step 1] Creating RFV clusterer...")
    clusterer = RFVClusterer()
    
    # Load sample data
    print("\n[Step 2] Loading sample data...")
    csv_path = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    df = pd.read_csv(csv_path, nrows=1000)  # Sample
    
    # Test clustering
    print("\n[Step 3] Testing clustering...")
    df_clustered = clusterer.fit_transform(df, "rfv1")
    
    # Show cluster distribution
    print("\n[Step 4] Cluster distribution:")
    cluster_counts = clusterer.get_cluster_counts(df, "rfv1")
    for cluster, count in cluster_counts.items():
        print(f"  {cluster}: {count}")
    
    # Show sample mappings
    print("\n[Step 5] Sample code-to-cluster mappings:")
    sample_codes = df['rfv1'].dropna().unique()[:10]
    for code in sample_codes:
        cluster = clusterer.get_cluster(float(code))
        code_text = clusterer.mapper.code_to_text(float(code), "rfv1")
        if code_text:
            label = code_text[:50] + "..." if len(code_text) > 50 else code_text
            print(f"  {code:.0f} -> {cluster}: {label}")
    
    # Save mapping
    print("\n[Step 6] Saving cluster mapping...")
    output_path = project_root / "data" / "rfv_cluster_mapping.json"
    clusterer.save_cluster_mapping(str(output_path))
    
    print("\n" + "=" * 80)
    print("CLUSTERING TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_clustering()

