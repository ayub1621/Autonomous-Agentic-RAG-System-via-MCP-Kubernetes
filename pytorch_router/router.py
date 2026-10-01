import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class IntentRouter:
    def __init__(self):
        print("Loading PyTorch embedding routing model (all-MiniLM-L6-v2)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load lightweight embedding model (~80MB)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
        self.model.eval()

        # Define domain anchor prototypes
        self.domain_prototypes = {
            "financial_data": [
                "stock market price, valuation, and ticker performance",
                "equity shares trading and recent market trend",
                "financial data, stock quotes, and corporate balance sheet"
            ],
            "crm_data": [
                "customer profile and enterprise account status",
                "client subscription details, MRR, and contracts",
                "internal database user record, account churn, and membership"
            ],
            "general_knowledge": [
                "geography, history, science, and world capitals",
                "general trivia and common factual explanations",
                "general conversational questions and broad topics"
            ]
        }
        
        # Precompute and cache normalized centroid vectors for each domain
        self.domain_centroids = {}
        with torch.no_grad():
            for domain, prototypes in self.domain_prototypes.items():
                proto_embs = self._embed(prototypes)
                # Compute centroid (mean vector) across prototypes and normalize
                centroid = proto_embs.mean(dim=0, keepdim=True)
                self.domain_centroids[domain] = F.normalize(centroid, p=2, dim=1)

    def _embed(self, texts: list[str]) -> torch.Tensor:
        """Tokenize, forward pass, mean-pool, and L2-normalize embeddings."""
        encoded = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model(**encoded)
        
        # Masked mean pooling over token embeddings
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        token_embeddings = outputs.last_hidden_state
        sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        return F.normalize(pooled, p=2, dim=1)

    def route_query(self, query: str) -> str:
        """Calculate cosine similarity between query embedding and domain centroids."""
        with torch.no_grad():
            query_emb = self._embed([query])
            similarities = {
                domain: F.cosine_similarity(query_emb, centroid).item()
                for domain, centroid in self.domain_centroids.items()
            }

        best_domain = max(similarities, key=similarities.get)
        confidence = similarities[best_domain]
        
        print(f"[{best_domain.upper()}] Cosine Sim: {confidence:.2f} | Query: '{query}'")
        return best_domain


if __name__ == "__main__":
    router = IntentRouter()
    print("\n--- Running Router Tests ---")
    test_queries = [
        "What is the current trend for Apple stock?",
        "Can you check if John Doe's enterprise account is active?",
        "What is the capital of France?"
    ]
    for q in test_queries:
        router.route_query(q)