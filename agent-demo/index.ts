import dotenv from "dotenv";
import * as path from "node:path";
import { OpenAI } from "openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import "cheerio";
import { pull } from "langchain/hub";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import type { ChatPromptTemplate } from "@langchain/core/prompts";

const envPath = path.resolve(__dirname, "./.env")

dotenv.config({
	path: envPath
})

const clientOptions = {
	baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
		apiKey: process.env.DASHSCOPE_API_KEY,
}

const main = async() => {

	// console.log('process.env.DASHSCOPE_API_KEY', process.env.DASHSCOPE_API_KEY)
	//
	// return

	const llm = new ChatOpenAI({
		model: "deepseek-r1-distill-llama-70b",
		temperature: 0,
		configuration: clientOptions
	});

	const embeddings = new OpenAIEmbeddings({
		model: "text-embedding-v3",
		configuration: clientOptions
	});

	const vectorStore = new MemoryVectorStore(embeddings);


	// Load and chunk contents of blog
	const pTagSelector = "p";
	const cheerioLoader = new CheerioWebBaseLoader(
		// "https://github.com/features/security/",
		"https://lilianweng.github.io/posts/2023-06-23-agent/",
		{
			selector: pTagSelector
		}
	);

	const docs = await cheerioLoader.load();

	const splitter = new RecursiveCharacterTextSplitter({
		chunkSize: 1000, chunkOverlap: 200
	});
	const allSplits = await splitter.splitDocuments(docs);

	// Index chunks
	await vectorStore.addDocuments(allSplits)

	// Define prompt for question-answering
	const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

	const InputStateAnnotation = Annotation.Root({
		question: Annotation<string>,
	});

	const StateAnnotation = Annotation.Root({
		question: Annotation<string>,
		context: Annotation<Document[]>,
		answer: Annotation<string>,
	});

	// Define application steps
	const retrieve = async (state: typeof InputStateAnnotation.State) => {
		const retrievedDocs = await vectorStore.similaritySearch(state.question)
		return { context: retrievedDocs };
	};

	const generate = async (state: typeof StateAnnotation.State) => {
		const docsContent = state.context.map(doc => doc.pageContent).join("\n");
		const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
		const response = await llm.invoke(messages);
		return { answer: response.content };
	};

	// Compile application and test
	const graph = new StateGraph(StateAnnotation)
		.addNode("retrieve", retrieve)
		.addNode("generate", generate)
		.addEdge("__start__", "retrieve")
		.addEdge("retrieve", "generate")
		.addEdge("generate", "__end__")
		.compile();

	let inputs = { question: "What is Task Decomposition?" };
	const result = await graph.invoke(inputs);
	console.log(result.answer);

}

main()

// const openai = new OpenAI(
// 	{
// 		// 若没有配置环境变量，请用百炼API Key将下行替换为：apiKey: "sk-xxx"
// 		apiKey: process.env.DASHSCOPE_API_KEY,
// 		baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1"
// 	}
// );
//
// (async () => {
// 	const completion = await openai.chat.completions.create({
// 		model: "deepseek-r1-distill-llama-70b",
// 		messages: [
// 			{ role: "system", content: "You are a helpful assistant." },
// 			{ role: "user", content: "9.9和9.11谁大" }
// 		],
// 	});
// 	console.log(completion.choices[0].message.content)
// })()



