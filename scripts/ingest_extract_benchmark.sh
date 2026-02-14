#! /bin/zsh

# Ingestion of howtos

: "${BENCHMARK_QUERY:=match_all}"

uv run theme-extractor ingest \
  --input "howto/" \
  --reset-index \
  --index-backend \
  --recursive \
  --cleaning-options all \
  --manual-stopwords "de,le,la,the,and,of" \
  --manual-stopwords-file src/theme_extractor/resources/stopwords_en_fallback.txt \
  --auto-stopwords \
  --auto-stopwords-min-doc-ratio 0.7 \
  --auto-stopwords-min-corpus-ratio 0.01 \
  --auto-stopwords-max-terms 200 \
  --pdf-ocr-fallback \
  --pdf-ocr-languages fra+eng \
  --pdf-ocr-dpi 200 \
  --pdf-ocr-min-chars 32 \
  --msg-include-metadata \
  --msg-attachments-policy names \
  --streaming-mode \
  --output data/out/ingest.json

# Extract themes using different methods and parameters for benchmarking

uv run theme-extractor extract \
  --method bertopic \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --bertopic-use-embeddings \
  --bertopic-embedding-model bge-m3 \
  --bertopic-dim-reduction umap \
  --bertopic-clustering hdbscan \
  --bertopic-min-topic-size 5 \
  --bertopic-embedding-cache-enabled \
  --bertopic-embedding-cache-dir data/cache/embeddings \
  --bertopic-embedding-cache-version v1 \
  --output data/out/extract_bertopic_umap_hdbscan.json

uv run theme-extractor extract \
  --method bertopic \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --bertopic-use-embeddings \
  --bertopic-embedding-model bge-m3 \
  --bertopic-dim-reduction nmf \
  --bertopic-clustering hdbscan \
  --bertopic-min-topic-size 5 \
  --bertopic-embedding-cache-enabled \
  --bertopic-embedding-cache-dir data/cache/embeddings \
  --bertopic-embedding-cache-version v1 \
  --output data/out/extract_bertopic_nmf_hdbscan.json

uv run theme-extractor extract \
  --method bertopic \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --bertopic-use-embeddings \
  --bertopic-embedding-model bge-m3 \
  --bertopic-dim-reduction svd \
  --bertopic-clustering hdbscan \
  --bertopic-min-topic-size 5 \
  --bertopic-embedding-cache-enabled \
  --bertopic-embedding-cache-dir data/cache/embeddings \
  --bertopic-embedding-cache-version v1 \
  --output data/out/extract_bertopic_svd_hdbscan.json

uv run theme-extractor extract \
  --method bertopic \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --bertopic-use-embeddings \
  --bertopic-embedding-model bge-m3 \
  --bertopic-dim-reduction umap \
  --bertopic-clustering kmeans \
  --bertopic-min-topic-size 5 \
  --bertopic-embedding-cache-enabled \
  --bertopic-embedding-cache-dir data/cache/embeddings \
  --bertopic-embedding-cache-version v1 \
  --output data/out/extract_bertopic_umap_kmeans.json

uv run theme-extractor extract \
  --method bertopic \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --bertopic-use-embeddings \
  --bertopic-embedding-model bge-m3 \
  --bertopic-dim-reduction nmf \
  --bertopic-clustering kmeans \
  --bertopic-min-topic-size 5 \
  --bertopic-embedding-cache-enabled \
  --bertopic-embedding-cache-dir data/cache/embeddings \
  --bertopic-embedding-cache-version v1 \
  --output data/out/extract_bertopic_nmf_kmeans.json

uv run theme-extractor extract \
  --method bertopic \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --bertopic-use-embeddings \
  --bertopic-embedding-model bge-m3 \
  --bertopic-dim-reduction svd \
  --bertopic-clustering kmeans \
  --bertopic-min-topic-size 5 \
  --bertopic-embedding-cache-enabled \
  --bertopic-embedding-cache-dir data/cache/embeddings \
  --bertopic-embedding-cache-version v1 \
  --output data/out/extract_bertopic_svd_kmeans.json

uv run theme-extractor extract \
  --method baseline_tfidf \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --output data/out/extract_tfidf.json

uv run theme-extractor extract \
  --method terms \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --output data/out/extract_terms.json

uv run theme-extractor extract \
  --method significant_terms \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "extract" \
  --output data/out/extract_significant_terms.json

uv run theme-extractor extract \
  --method significant_text \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "extract" \
  --output data/out/extract_significant_text.json

uv run theme-extractor extract \
  --method llm \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "match_all" \
  --offline-policy preload_or_first_run \
  --llm-provider openai \
  --llm-api-base-url https://litellm.g1lom.xyz/v1 \
  --llm-model "gpt-5-nano" \
  --output data/out/extract_llm.json

uv run theme-extractor extract \
  --method keybert \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --bertopic-embedding-model bge-m3 \
  --query "match_all" \
  --keybert-use-embeddings \
  --fields content,filename,path \
  --source-field content \
  --topn 25 \
  --search-size 200 \
  --output data/out/extract_keybert.json

# Benchmarking with different parameters for BERTopic

uv run theme-extractor benchmark \
  --methods baseline_tfidf,terms,keybert,bertopic,llm \
  --backend elasticsearch \
  --backend-url http://localhost:9200 \
  --index theme_extractor \
  --focus both \
  --query "${BENCHMARK_QUERY}" \
  --bertopic-min-topic-size 5 \
  --search-size 200 \
  --output data/out/benchmark_all.json

uv run theme-extractor evaluate \
  --input data/out/benchmark_all.json \
  --output data/out/evaluation_benchmark_all.json

# Reporting MD

uv run theme-extractor report \
  --input data/out/extract_tfidf.json \
  --output data/out/report_extract_tfidf.md

uv run theme-extractor report \
  --input data/out/extract_terms.json \
  --output data/out/report_extract_terms.md

uv run theme-extractor report \
  --input data/out/extract_keybert.json \
  --output data/out/report_extract_keybert.md

uv run theme-extractor report \
  --input data/out/extract_llm.json \
  --output data/out/report_extract_llm.md

uv run theme-extractor report \
  --input data/out/extract_bertopic_svd_kmeans.json \
  --output data/out/report_extract_bertopic_svd_kmeans.md

uv run theme-extractor report \
  --input data/out/extract_bertopic_nmf_kmeans.json \
  --output data/out/report_extract_bertopic_nmf_kmeans.md

uv run theme-extractor report \
  --input data/out/extract_bertopic_umap_kmeans.json \
  --output data/out/report_extract_bertopic_umap_kmeans.md

uv run theme-extractor report \
  --input data/out/extract_bertopic_svd_hdbscan.json \
  --output data/out/report_extract_bertopic_svd_hdbscan.md

uv run theme-extractor report \
  --input data/out/extract_bertopic_nmf_hdbscan.json \
  --output data/out/report_extract_bertopic_nmf_hdbscan.md

uv run theme-extractor report \
  --input data/out/extract_bertopic_umap_hdbscan.json \
  --output data/out/report_extract_bertopic_umap_hdbscan.md

uv run theme-extractor report \
  --input data/out/benchmark_all.json \
  --title "Benchmark Report - Howtos from theme-extractor" \
  --output data/out/report_benchmark.md

uv run theme-extractor report \
  --input data/out/evaluation_benchmark_all.json \
  --title "Evaluation Report - Howtos from theme-extractor" \
  --output data/out/report_evaluation_benchmark_all.md
