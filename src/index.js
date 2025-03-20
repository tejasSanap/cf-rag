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

const google = createGoogleGenerativeAI({
	apiKey: '',
});

app.post('/mrt', async (c) => {
	try {
		const body = await c.req.json();
		const { prompt, config } = body;
		console.log('prompt', prompt);
		console.log('config', config);

		const systemPrompt1 = `
		You are an AI assistant called Superpumped that processes table data for a Material React Table (MRT) V2 component.
		Given the table data and a user prompt, interpret the prompt and return a JSON object representing the updated MRT configuration.
		The configuration must include:
		- "data": the updated table data (array of objects),
		- "columns": the updated column definitions (array of {accessorKey, header}, optionally with other MRT props like enableColumnOrdering),
		- Optionally, other MRT props like "initialState" for sorting/filtering/configuration.
		Be concise and accurate. If the prompt is unclear, return an error message in the JSON.
		Example input: Data: [{"firstName": "John", "lastName": "Doe"}], Prompt: "remove column lastName"
		Example output: {"data": [{"firstName": "John"}], "columns": [{"accessorKey": "firstName", "header": "First Name"}]}
		`;

		const systemPrompt = `You are an AI assistant called Superpumped that processes table data for a Material React Table (MRT) V2 component.
Given the current table configuration and a user prompt, interpret the prompt and return a JSON object representing the updated MRT configuration.
The configuration must always include:
- "data": the updated table data (array of objects),
- "columns": the updated column definitions (array of {accessorKey, header}, optionally with MRT props like enableColumnOrdering),
- Other MRT props as needed (e.g., enableGrouping, enablePagination, initialState) based on the prompt.

Key MRT V2 configuration details:
- "enableGrouping": boolean to enable/disable grouping feature (default: false).
- "initialState": object to set default table state, e.g., { grouping: ['columnKey'] } to group by specific columns by default.
- "enableColumnOrdering": boolean to allow column reordering.
- "enablePagination": boolean to enable/disable pagination.

Rules:
- If the prompt modifies a feature (e.g., "enable grouping"), explicitly set the corresponding prop (e.g., "enableGrouping": true).
- If the prompt specifies grouping by a column (e.g., "group by age"), set "enableGrouping": true and "initialState": { grouping: ['age'] }.
- If the prompt removes a column, update both "data" and "columns" accordingly.
- If the prompt is unclear, return { "error": "Unclear prompt, please specify the action" }.
- Preserve existing config props unless the prompt explicitly changes them.

Examples:
1. Input: Config: { "data": [{"firstName": "John", "lastName": "Doe"}], "columns": [{"accessorKey": "firstName", "header": "First Name"}, {"accessorKey": "lastName", "header": "Last Name"}] }, Prompt: "remove column lastName"
   Output: { "data": [{"firstName": "John"}], "columns": [{"accessorKey": "firstName", "header": "First Name"}] }
2. Input: Config: { "data": [{"age": 30}], "columns": [{"accessorKey": "age", "header": "Age"}], "enableGrouping": false }, Prompt: "enable grouping"
   Output: { "data": [{"age": 30}], "columns": [{"accessorKey": "age", "header": "Age"}], "enableGrouping": true }
3. Input: Config: { "data": [{"age": 30}], "columns": [{"accessorKey": "age", "header": "Age"}], "enableGrouping": false }, Prompt: "group by age"
   Output: { "data": [{"age": 30}], "columns": [{"accessorKey": "age", "header": "Age"}], "enableGrouping": true, "initialState": { "grouping": ["age"] } }

Be concise, accurate, and return valid JSON.`;
		const userMessage = `
			Table data and config: ${JSON.stringify(config)}
			User prompt: "${prompt}"
		`;

		const initialMessages = [{ role: 'user', content: systemPrompt }];
		const userPromptMessage = { role: 'user', content: userMessage };

		// Call Gemini API
		const response = await generateText({
			model: google('gemini-1.5-flash'),
			messages: [...initialMessages, userPromptMessage],
		});

		// let resultText = '';
		// for await (const chunk of response) {
		// 	resultText += chunk;
		// }
		const resultText = response.text || response.data; // Adjust based on your API response structure

		// Clean up the response to remove ```json and ```
		const cleanedText = resultText
			.replace(/```json/g, '') // Remove opening ```json
			.replace(/```/g, '') // Remove closing ```
			.trim(); // Remove leading/trailing whitespace

		console.log('Cleaned text:', cleanedText);
		let updatedConfig;
		// return c.text('Lets see');
		// Parse the response as JSON
		try {
			updatedConfig = JSON.parse(cleanedText);
		} catch (parseError) {
			console.error('Failed to parse Gemini response:', parseError);
			return c.status(500).json({ error: 'Invalid response format from AI' });
		}
		console.log('update config, ', updatedConfig);

		// Validate the response structure
		if (!updatedConfig.data || !updatedConfig.columns) {
			return c.status(400).json({ error: 'Invalid configuration: missing data or columns' });
		}

		// Send the updated configuration back to the frontend
		return c.json({ updatedConfig });
	} catch (error) {
		// Handle any errors and send a response back to the client
		console.error('Error:', error);
		return res.status(500).json({ error: 'Failed to process the request' });
	}
});

app.post('/mrt-chat', async (c) => {
	try {
		const body = await c.req.json();
		console.log('body', body);

		const { data, prompt } = body;

		// Validate input
		if (!prompt || !data || !Array.isArray(data)) {
			return c.text('Prompt and data are required, and data must be an array', 400);
		}

		// Format the data as a CSV-like string for better readability by the model
		let formattedData = '';
		if (Array.isArray(data) && data.length > 0) {
			// Extract column headers from the first row
			const headers = Object.keys(data[0]);
			// Add headers as the first row
			formattedData = headers.join(', ') + '\n';
			// Add each row of data
			formattedData += data.map((row) => headers.map((header) => row[header] ?? '').join(', ')).join('\n');
		} else {
			formattedData = 'No data available.';
		}

		// Improved system prompt with detailed instructions and examples
		const systemPrompt = `
		You are an AI assistant called Superpumped that answers conversational queries about table data for a Material React Table (MRT) V2 component.
		The table data is provided in a CSV-like format, where the first row contains the column headers, and each subsequent row represents a data entry.
		Your task is to interpret the user's prompt and provide a concise, natural language response based on the data.
  
		Key Instructions:
		- The data is formatted as: "header1, header2, header3\nvalue1, value2, value3\n..."
		- Analyze the data to answer queries about highest/lowest values, averages, counts, filtering, or other aggregations.
		- Respond in a conversational tone, as if speaking to the user directly.
		- If the prompt is unclear or the query cannot be answered with the given data, respond with: "I couldn't understand your query or the data doesn't support this request. Please try again."
		- Do not modify the data or table configuration; only provide a textual response.
		- If a column name in the prompt doesn't match the data, try to infer the closest match (e.g., "sales" might be "Sales").
		- Handle numerical comparisons (e.g., "above", "below") and aggregations (e.g., "average", "total") accurately.
		- For queries involving names, use both firstName and lastName if available (e.g., "John Doe").
  
		Examples:
		1. Data: "firstName, lastName, sales\nJohn, Doe, 50000\nJane, Smith, 75000"
		   Prompt: "who has the highest sales?"
		   Response: "Jane Smith has the highest sales with a value of 75000."
		2. Data: "firstName, lastName, salary, department\nJohn, Doe, 60000, Sales\nJane, Smith, 80000, Sales\nBob, Johnson, 55000, Marketing"
		   Prompt: "average salary in each department"
		   Response: "Average salary in Sales: 70000. Average salary in Marketing: 55000."
		3. Data: "firstName, lastName, region\nJohn, Doe, North\nJane, Smith, South\nBob, Johnson, North"
		   Prompt: "how many people in each region?"
		   Response: "There are 2 people in North and 1 person in South."
		4. Data: "firstName, lastName, salary, department\nJohn, Doe, 60000, Sales\nJane, Smith, 80000, Sales"
		   Prompt: "list people in Sales with salary above 70000"
		   Response: "Jane Smith has a salary of 80000 in Sales."
		5. Data: "firstName, lastName, age\nJohn, Doe, 30\nJane, Smith, 25"
		   Prompt: "who is the youngest?"
		   Response: "Jane Smith is the youngest with an age of 25."
  
		Now, analyze the following data and answer the user's prompt.
	  `;

		const content = `
		The following table data is provided for analysis:
  
		${formattedData}
  
		Now, please answer the following question:
	  `;

		const initialMessages = [{ role: 'user', content: systemPrompt }];
		const userMessage = { role: 'user', content: `${content}\n${prompt}` };

		const response = await streamText({
			model: google('gemini-1.5-flash'),
			messages: [...initialMessages, userMessage],
		});

		// Handle the response as a stream
		const r = response.toTextStreamResponse();

		return r;
	} catch (error) {
		console.error('Error during processing:', error);
		return c.text('Failed to process the request', 500);
	}
});

// app.post('/mrt-chat', async (c) => {
// 	try {
// 		const body = await c.req.json();
// 		console.log('body', body);
// 		const { data, prompt } = body;
// 		// const systemPrompt = `You are an AI assistant called superpumped that acts as a "Second Brain" by answering questions based on provided context. Your goal is to directly address the question concisely and to the point, without excessive elaboration`;

// 		// systemPrompt: "${systemPrompt}"

// 		const content = `
// 		Data : ${JSON.stringify(data)}
// 		`;
// 		const initialMessages = [{ role: 'user', content: content }];

// 		const userMessage = { role: 'user', content: prompt };

// 		const response = await streamText({
// 			model: google('gemini-1.5-flash'),
// 			messages: [...initialMessages, userMessage],
// 		});

// 		// let apiUrl = url.searchParams.get('apiurl');
// 		// response.headers.set('Access-Control-Allow-Origin', url.origin);
// 		const r = response.toTextStreamResponse();

// 		return r;
// 	} catch (error) {
// 		// Handle any errors and send a response back to the client
// 		console.error('Error:', error);
// 		return c.text('Failed to process the request', 500);
// 	}
// });

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
