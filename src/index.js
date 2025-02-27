import { Hono } from 'hono';
import { render } from 'hono/jsx/dom';
import { cors } from 'hono/cors';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { generateText, streamText, tool } from 'ai';

const app = new Hono();
app.use(
	'/*',
	cors({
		origin: '*', // Allow all origins
	})
);

app.get('/', async (c) => {
	const question = c.req.query('text') || 'What is the square root of 9?';
	const embeddings = await c.env.AI.run('@cf/baai/bge-base-en-v1.5', { text: question });
	const vectors = embeddings.data[0];
	const vectorQuery = await c.env.VECTOR_INDEX.query(vectors, { topK: 1 });

	let vecId;
	if (vectorQuery.matches && vectorQuery.matches.length > 0 && vectorQuery.matches[0]) {
		vecId = vectorQuery.matches[0].id;
	} else {
		console.log('No matching vector found or vectorQuery.matches is empty');
	}

	let notes = [];
	if (vecId) {
		const query = `SELECT * FROM notes WHERE id = ?`;
		const { results } = await c.env.DB.prepare(query).bind(vecId).all();
		if (results) notes = results.map((vec) => vec.text);
	}

	const contextMessage = notes.length ? `Context:\n${notes.map((note) => `- ${note}`).join('\n')}` : '';

	const systemPrompt = `When answering the question or responding, use the context provided, if it is provided and relevant.`;

	const { response: answer } = await c.env.AI.run('@cf/meta/llama-3-8b-instruct', {
		messages: [
			...(notes.length ? [{ role: 'system', content: contextMessage }] : []),
			{ role: 'system', content: systemPrompt },
			{ role: 'user', content: question },
		],
	});

	return c.text(answer);
});
app.post('/notes', async (c) => {
	const { text } = await c.req.json();
	console.log('text', text);
	if (!text) return c.text('missing text', 400);

	try {
		// Create database record
		const query = 'INSERT INTO notes (text) VALUES (?) RETURNING *';
		const { results } = await c.env.DB.prepare(query).bind(text).run();
		const record = results[0];
		if (!record) throw new Error('Failed to create note');

		// Generate embedding
		const embeddings = await c.env.AI.run('@cf/baai/bge-base-en-v1.5', { text: text });
		const values = embeddings.data[0];
		if (!values) throw new Error('Failed to generate vector embedding');

		// Insert vector
		await c.env.VECTOR_INDEX.upsert([
			{
				id: record.id.toString(),
				values: values,
			},
		]);

		return c.text('Created note', 201);
	} catch (e) {
		console.error(e);
		return c.text('Error creating note', 500);
	}
});

app.get('/api/chat', async (c) => {
	try {
		const query = c.req.query('query');
		console.log('query', query);
		const google = createGoogleGenerativeAI({
			apiKey: '',
		});
		const systemPrompt = `You are an AI assistant called superpumped that acts as a "Second Brain" by answering questions based on provided context. Your goal is to directly address the question concisely and to the point, without excessive elaboration`;

		const initialMessages = [
			{ role: 'user', content: systemPrompt },
			{ role: 'assistant', content: 'Hello, how can I help?' },
		];
		const userMessage = { role: 'user', content: query };
		const response = await streamText({
			model: google('gemini-1.5-flash'),
			messages: [...initialMessages, userMessage],
		});

		// let apiUrl = url.searchParams.get('apiurl');
		// response.headers.set('Access-Control-Allow-Origin', url.origin);
		const r = response.toTextStreamResponse();
		return r;
	} catch (error) {
		// Handle any errors and send a response back to the client
		console.error('Error:', error);
		return c.text('Failed to process the request', 500);
	}
});

// app.post('/notes', async (c) => {
// 	const { text } = await c.req.json();
// 	console.log('text', text);
// 	if (!text) return c.text('missing text', 400);

// 	try {
// 		const r = await c.env.RAG_WORKFLOW.create({ params: { text } });
// 		console.log('r', r);
// 		return c.text('Created note', 201);
// 	} catch (e) {
// 		console.error(e);
// 	}
// });

export class RAGWorkflow {
	async run(event, step) {
		console.log('evv', event);
		const { text } = event.params;

		const record = await step.do(`create database record`, async () => {
			const query = 'INSERT INTO notes (text) VALUES (?) RETURNING *';

			const { results } = await env.DATABASE.prepare(query).bind(text).run();

			const record = results[0];
			if (!record) throw new Error('Failed to create note');
			return record;
		});
		console.log('record', record);
		const embedding = await step.do(`generate embedding`, async () => {
			const embeddings = await env.AI.run('@cf/baai/bge-base-en-v1.5', { text: text });
			const values = embeddings.data[0];
			if (!values) throw new Error('Failed to generate vector embedding');
			return values;
		});
		console.log('embedding', embedding);
		await step.do(`insert vector`, async () => {
			return env.VECTOR_INDEX.upsert([
				{
					id: record.id.toString(),
					values: embedding,
				},
			]);
		});
	}
}

export default app;
