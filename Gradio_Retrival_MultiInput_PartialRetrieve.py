# Fake 実験ノート by Noriさん 11-05
# ３つの入力領域を別々のクエリで検索する
# VectorStore: 各sectionに対して section単位でクエリ

# GradioでWebApp化する
# usage: "python .\Gradio_Retrival_MultiInput_Partial.py"
#        URL http://0.0.0.0:7860
#        Use Ctrl+C on console to terminate the server,

# pip install gradio          # WebApp用
# pip install google-genai    # embeddings用
# pip install chromadb        # vector store, similarity search用
# pip install cohere          # Re-ranikn用

from pprint import pprint
import re, glob, json
import pandas as pd
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 20)

import gradio as gr
import google.genai as genai    # embedding用
import chromadb                 # Vector Store用
import cohere                   # Re-raniking用

# Geminiモデルを指定
GEMINI_EMBEDDING_MODEL = 'gemini-embedding-001'
#GEMINI_LLM_MODEL      = 'gemini-2.0-flash'
# Cohereモデルを指定
#COHERE_EMBEDDING_MODEL = 'embed-v4.0'
#COHERE_LLM_MODEL       = 'command-a-03-2025'
COHERE_RERANK_MODEL     = 'rerank-v3.5'

def initialize_clients():
    """クライアントの初期化"""

     # Geminiクライアントを作成
    with open('GOOGLE_API_KEY.txt', 'r') as f:  # ファイルからアクセスキーを取得
        api_key = f.read().strip()
    gemini_client = genai.Client(api_key=api_key)

    # --- Chromaクライアントを作成
    chroma_client = chromadb.EphemeralClient()  # インメモリで作成

    # Cohereクライアントを作成
    with open('Cohere_API_KEY.txt', 'r') as f:  # ファイルからアクセスキーを取得
        api_key = f.read().strip()
    co = cohere.ClientV2(api_key=api_key)

    return gemini_client, chroma_client, co

def read_notes(doc_dir='./'):
    """
    Note(json)の読み込み
    戻り値: 辞書
            {title1:{'objective':..., 'materials':..., 'procedure':..., ... }, ...  }
    """

    # doc_dir以下のすべての .json ファイルを取得
    json_files = glob.glob(f"{doc_dir}*.json")
    print(json_files)
    # レシピを辞書に格納
    notes = []
    for i, file_path in enumerate(json_files, start=1):
        with open(file_path, "r", encoding="utf-8") as f:
            in_notes = json.load(f)
            notes += in_notes
    
    #pprint(notes)
    print(f"\ndoc_dir={doc_dir}, 読みこんだレシピの数: {len(notes)}")
    
    notes_dic = {}
    for i, r_dic in enumerate(notes):
        if title := r_dic.get('title', False):
            notes_dic[title] = r_dic
    #pprint(notes_dic)
    
    return notes_dic

def setup_collection(_notes_dic, _gemini_client, _chroma_client):
    """コレクションのセットアップ"""

    # collection作成
    collection_name = 'notes'
    collection = _chroma_client.create_collection(
        name = collection_name,
        metadata={'hnsw:space': 'cosine'}  # 距離メトリック = 'cosine'
    )

    titles = list(_notes_dic.keys())
    n = 30 # 一度に処理すレシピ数（x section数 = バッチサイズ）
    for i in range(0, len(titles), n): # バッチ毎に
        page_contents = []
        metadatas     = []
        ids           = []

        for j in range(i, min(i+n, len(titles))): # ノート毎に
            title = titles[j]
            note = _notes_dic[title]
            for k, s in enumerate(['objective','materials','procedure']): # section毎に
                page_contents.append(f"## {s}: \n{note[s]}")
                metadatas.append({'source': title, 'section': s})
                ids.append(f"doc{j}_{k}")    # id: ユニークな文字列

        # --- Embeddingの取得 ---
        doc_embs = get_embeddings(page_contents, _gemini_client, GEMINI_EMBEDDING_MODEL)
        
        # ChromaDBへ一括で追加
        collection.add(
            ids=ids,
            embeddings=doc_embs,
            documents=page_contents,
            metadatas=metadatas
        )
    print(f"DBに追加されたベクトル数: {collection.count()}")

    return collection


### 共通のEmbedding関数 =============
def get_embeddings(texts, client, embedding_model):
    # 単一文字列でもリストでも対応
    if isinstance(texts, str):
        texts = [texts]
        single_input = True
    else:
        single_input = False
    
    # Embedding取得
    response = client.models.embed_content(
        model=embedding_model,
        contents=texts
    )
    embeddings = [e.values for e in response.embeddings]
    
    # 単一入力の場合は1次元リストを返す
    if single_input:
        return embeddings[0]
    
    return embeddings

### Vector Store(ChromaDB)のsection毎に類似検索(Retrive) =============
def retrieve(queries_dic, k=10): # queries_dic
    print(f"### Retrieve: query=\n「{queries_dic}」, k={k} ") 

    # section毎にフィルタをかけてRetrieve, rerank
    reranked_df = pd.DataFrame({
        'docs':   pd.Series(dtype=str),
        'source': pd.Series(dtype=str),
        'score':  pd.Series(dtype=float)
    })
    n_queries = 0
    for i, section in enumerate(['objective','materials','procedure']):
        query = queries_dic[section]
        if query == '':
            continue
        
        n_queries += 1
        # --- クエリをembedding ---
        query_emb = get_embeddings(query, gemini_client, GEMINI_EMBEDDING_MODEL)
    
        # ChromaDBで類似検索（＝retrieval）=======
        results = collection.query(
            query_embeddings=query_emb,
            n_results = 4*k,
            include = ['documents', 'metadatas', 'distances'],
            where = {'section': section}                        # metadataでのfilter条件
        )
        retreaved_docs    = results['documents'][0]
        retreaved_sources = [m['source'] for m in results['metadatas'][0]]
        retreaved_dists   = results['distances'][0]
        retreaved_df = pd.DataFrame({
            'docs' :  retreaved_docs,
            'source': retreaved_sources,
            'dists':  retreaved_dists
        })
        #print(retreaved_df)
        
        # Cohereでdocumentsをリランク(Rerank) =============
        results = co.rerank(
            model=COHERE_RERANK_MODEL,
            query=query,
            documents=retreaved_docs,
            top_n=4*k,
        ).results

        reranked_df = pd.concat(  # section毎のrerankを連結
            [reranked_df,
             pd.DataFrame({
                'docs':   [retreaved_docs[r.index]    for r in results],
                'source': [retreaved_sources[r.index] for r in results],
                'score':  [r.relevance_score          for r in results]
            })],
            axis=0, ignore_index=True)
    aggr_reranked_df = (
        reranked_df.groupby('source', as_index=False)['score'].sum()  # sourceで集約, スコアは合計
        .sort_values('score', ascending=False).reset_index(drop=True) # scoreでソート
    ).head(k)
    aggr_reranked_df['score'] = aggr_reranked_df['score'] / n_queries # Queryの数で割り戻す

    #print(reranked_df)
    #print(aggr_reranked_df)

    return aggr_reranked_df[['source','score']].values.tolist()


# 初期設定 ==============================================
# クライアント定義とデータ読み込み、コレクションを作成

gemini_client, chroma_client, co  = initialize_clients()                   # clientの定義
recipes_dic = read_notes()                                                 # レシピの読み込み
collection = setup_collection(recipes_dic, gemini_client, chroma_client)   # collectionの作成


# 入力画面 [Gradio 依存部分] ============================================================

def search_notes(input1, input2, input3, history):

    input1_clean = input1.strip()
    input2_clean = input2.strip()
    input3_clean = input3.strip()

    if not any([input1_clean, input2_clean, input3_clean]):
        history.append({
            'role': 'assistant', 'content': "⚠️ 入力が空です。少なくとも1つ入力してください。"
        })
        return history

    queries_dic = {'objective': input1_clean, 'materials': input2_clean, 'procedure': input3_clean}

    # 🆕 ユーザーメッセージを履歴に追加（整形版）
    user_display = f"""
**検索条件:**
- 目的: {input1_clean if input1_clean else '指定なし'}
- 試薬: {input2_clean if input2_clean else '指定なし'}
- 手順: {input3_clean if input3_clean else '指定なし'}
"""
    history.append({'role': 'user', 'content': user_display})  # 入力内容の表示

    try:
        response = retrieve(queries_dic, k=10)
        # 結果を整形
        response_texts = '**検索結果:** (Rank: Title, Score)  \n' +\
            '  \n'.join([f"{i:2d}: {p}, {s:.3f}" for i, (p, s) in enumerate(response, start=1)])
    
    except Exception as e:
        response_texts = f"❌ エラーが発生しました: {str(e)}"

    print(response_texts)
    history.append({'role': 'assistant', 'content': response_texts})  # 検索結果の表示

    return history

def clear_inputs():
    return ['', '', '']

# Gradio UI構築
with gr.Blocks(title="🍳 レシピ検索") as search:
    gr.Markdown("## 🍳 レシピ検索")
    gr.Markdown("### 検索条件を入力してください")
    
    with gr.Row():
        with gr.Column(scale=1): # 左ペイン, 幅1
            input1 = gr.Textbox(label="目的", placeholder="例: タンパク質を測定する", lines=2)
            input2 = gr.Textbox(label="試薬", placeholder="例: BSA, ビウレット試薬", lines=2)
            input3 = gr.Textbox(label="手順", placeholder="例: 混合して吸光度を測定", lines=2)
            
            with gr.Row():
                search_btn = gr.Button("🔍 検索", variant="primary")
                clear_btn  = gr.Button("🗑️ 入力クリア")
        
        with gr.Column(scale=2): # 右ペイン, 幅3
            chatbot = gr.Chatbot(
                label="検索履歴と結果",
                height=600,
                show_label=True,
                type='messages'
            )
            clear_history_btn = gr.Button("🗑️ 履歴クリア")

    # イベントハンドラ
    search_btn.click(
        fn=search_notes, inputs=[input1, input2, input3, chatbot], outputs=chatbot
    )

    clear_btn.click(
        fn=clear_inputs, inputs=None, outputs=[input1, input2, input3]
    )

    clear_history_btn.click(
        fn=lambda: [], inputs=None, outputs=chatbot
    )

# アプリ起動
if __name__ == "__main__":
    search.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )