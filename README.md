# 🔗 HMTC-Refinement: Hierarchical Text Classification
### **Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification**

---

## 📝 Overview
This project implements a weakly-supervised hierarchical text classification (HMTC) method. It combines the structural knowledge of taxonomies with the analytical power of **Google Gemini 2.5**. 

---

## 🏗️ System Architecture

> ### **Step 1: Silver Label Generation**
> * **Hybrid Retrieval**: Combines **SBERT** (Semantic) and **BM25** (Lexical) for initial matching.
> * **Path Restoration**: Restores full taxonomy paths from leaf-node candidates.
> * **Probability Normalization**: Applies sibling-normalized softmax for optimal path depth.

---

> ### **Step 2: Taxonomy-Aware Training**
> * **Document Encoder**: BERT-based frozen transformer for robust text representation.
> * **Class Encoder (GNN)**: A Graph Convolutional Network that refines class embeddings across the hierarchy.
> * **Dual-Loss Objective**: Combines **Taxonomy-Aware BCE** and **Supervised Contrastive Loss**.

---

> ### **Step 3: Selective LLM Refinement**
> * **Uncertainty Scoring**: Identifies ambiguous samples using *Margin* and *Leaf Gap* metrics.
> * **Budgeted Gemini API**: Only high-uncertainty samples are sent to **Gemini-1.5-Flash**.
> * **Candidate Prompting**: Provides the LLM with narrowed candidates for precise selection.

---

## 📊 Performance Results
| Method Variant | Micro-F1 Score |
| :--- | :---: |
| Node-wise Prediction (Baseline) | 0.37 |
| Constrained Path Decoding | 0.53 |
| **Selective LLM Refinement (Final)** | **0.60** |

---

## 🚀 How to Run (실행 방법)

### 1. Environment Setup (환경 설정)
LMS에 제출된 폴더 내의 `.env` 파일을 아래 구조와 같이 최상위 루트 폴더(Root Directory)에 위치시켜 주세요. `.env` 파일이 `src` 폴더 내부가 아닌 **바깥쪽**에 있어야 프로그램이 정상적으로 API 키를 로드할 수 있습니다.

```text
20252R0136DATA30400 (Root)
├── .env                 <-- [중요: 여기에 위치시켜 주세요]
├── main.py              <-- [실행 파일]
├── requirements.txt     <-- [의존성 목록]
└── src/                 
    └── trainer.py       <-- (소스 코드 폴더)
'''

### 2. Install Dependencies**
'''bash
pip install -r requirements.txt
'''

### 3. Execution
'''bash
python main.py
'''
