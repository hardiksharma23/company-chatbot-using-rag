import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { pipeline } from "@huggingface/transformers";
import { Pinecone } from "@pinecone-database/pinecone";

const embeddingModel = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
);

export async function indexTheDocument(filePath) {
    const loader = new PDFLoader(filePath, { splitPages: false });
    const doc = await loader.load();

    // console.log(doc[0].pageContent);

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 100,
    });

    const texts = await splitter.splitText(doc[0].pageContent);

    // console.log(texts.length);

    const embeddings = [];

    for (const text of texts) {
        const output = await embeddingModel(text, {
            pooling: "mean",
            normalize: true,
        });

        embeddings.push(Array.from(output.data));
    }

    const vectors = embeddings.map((embedding, index) => ({
        id: `chunk-${index}`,
        values: embedding,
        metadata: {
            text: texts[index]
        }
    }));

    // console.log("Vectors prepared:", vectors.length);

    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    });

    const index = pc.index(process.env.PINECONE_INDEX_NAME);

    await index.upsert({
        records: vectors,
    });

    console.log("Vectors uploaded to Pinecone successfully!");

    // console.log("Number of embeddings:", embeddings.length);
    // console.log("Embedding dimension:", embeddings[0].length);
    // console.log("First embedding:", embeddings[0]);

}