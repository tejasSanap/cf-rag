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
		console.log('formattedData', formattedData);
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

app.post('/chart-config-2', async (c) => {
	try {
		const body = await c.req.json();
		console.log('body', body);

		const { data, prompt, config } = body;

		// Validate input
		if (!prompt || !data || !Array.isArray(data)) {
			return c.text('Prompt and data are required, and data must be an array', 400);
		}

		// Format data into CSV-like structure
		let formattedData = '';
		if (data.length > 0) {
			const headers = Object.keys(data[0]);
			formattedData = headers.join(', ') + '\n' + data.map((row) => headers.map((header) => row[header] ?? '').join(', ')).join('\n');
		} else {
			formattedData = 'No data available.';
		}

		// System instructions for chart configuration update
		const systemPrompt = `You are an AI assistant that updates ApexCharts configurations based on user requests...`;
		const content = `The following table data is provided:\n\n${formattedData}\n\nNow, update the chart configuration based on this prompt:\n${prompt}`;

		const response = await generateText({
			model: google('gemini-1.5-flash'),
			messages: [
				{ role: 'user', content: systemPrompt },
				{ role: 'user', content: content },
			],
		});

		const cleanedText = response.text
			.replace(/```json/g, '')
			.replace(/```/g, '')
			.trim();
		let updatedChartConfig;
		try {
			updatedChartConfig = JSON.parse(cleanedText);
		} catch (parseError) {
			console.error('Failed to parse chart config:', parseError);
			return c.text('Invalid chart configuration from AI', 500);
		}

		if (!updatedChartConfig.options || !updatedChartConfig.series) {
			return c.text('Invalid chart configuration: missing options or series', 400);
		}

		return c.json({ updatedChartConfig });
	} catch (error) {
		console.error('Error in /mrt-config:', error);
		return c.text('Failed to process the request', 500);
	}
});
app.post('/chart-chat', async (c) => {
	try {
		const body = await c.req.json();
		console.log('body', body);

		const { data, prompt } = body;

		// Validate input
		if (!prompt || !data || !Array.isArray(data)) {
			return c.text('Prompt and data are required, and data must be an array', 400);
		}

		// Format data into CSV-like structure
		let formattedData = '';
		if (data.length > 0) {
			const headers = Object.keys(data[0]);
			formattedData = headers.join(', ') + '\n' + data.map((row) => headers.map((header) => row[header] ?? '').join(', ')).join('\n');
		} else {
			formattedData = 'No data available.';
		}

		// System instructions for chat mode
		const systemPrompt = `You are an AI assistant that answers questions based on structured table data...`;
		const content = `The following table data is provided:\n\n${formattedData}\n\nNow, answer this question:\n${prompt}`;

		// Stream AI response
		const response = await streamText({
			model: google('gemini-1.5-flash'),
			messages: [
				{ role: 'user', content: systemPrompt },
				{ role: 'user', content: content },
			],
		});

		return response.toTextStreamResponse();
	} catch (error) {
		console.error('Error in /mrt-chat:', error);
		return c.text('Failed to process the request', 500);
	}
});

app.post('/chart-config', async (c) => {
	try {
		const body = await c.req.json();
		console.log('body', body);

		const { data, prompt, config } = body;

		// Validate input
		if (!prompt || !data || !Array.isArray(data)) {
			return c.text('Prompt and data are required, and data must be an array', 400);
		}

		// Format the data as a CSV-like string for better readability by the model
		let formattedData = '';
		if (Array.isArray(data) && data.length > 0) {
			const headers = Object.keys(data[0]);
			formattedData = headers.join(', ') + '\n';
			formattedData += data.map((row) => headers.map((header) => row[header] ?? '').join(', ')).join('\n');
		} else {
			formattedData = 'No data available.';
		}

		// Determine the mode based on whether config is provided
		const isConfigMode = !!config;

		if (isConfigMode) {
			// Config Mode: Update the chart configuration (non-streaming)
			const systemPrompt = `
		  You are an AI assistant called Superpumped that generates and updates chart configurations for ApexCharts based on table data and user prompts.
		  The table data is provided in a CSV-like format, where the first row contains the column headers, and each subsequent row represents a data entry.
		  The current chart configuration is provided in JSON format.
		  Your task is to interpret the user's prompt and return a JSON object representing the updated ApexCharts configuration, including:
		  - "options": The chart options (e.g., chart type, xaxis, yaxis, labels).
		  - "series": The data series for the chart (e.g., [{ name: "Sales", data: [50000, 75000] }]).
		  
		  Key Instructions:
		  - The data is formatted as: "header1, header2, header3\nvalue1, value2, value3\n..."
		  - The current chart config is: ${JSON.stringify(config)}.
		  - Determine the appropriate chart type based on the prompt (e.g., "bar chart" → bar, "pie chart" → pie, "line chart" → line).
		  - If the chart type is not specified, keep the current chart type or default to a bar chart for numerical data grouped by a categorical column.
		  - Aggregate data as needed (e.g., sum, average, count) based on the prompt.
		  - For bar charts, use a categorical column (e.g., department) for the x-axis and a numerical column (e.g., sales) for the y-axis.
		  - For pie charts, use a categorical column for labels and a numerical column for values.
		  - For line charts, ensure there’s a time-based or sequential column for the x-axis.
		  - If the prompt is unclear or the data doesn’t support the requested chart, return the current chart config with a message in the response.
		  - Preserve existing chart options unless the prompt explicitly changes them.
		  - Return a valid ApexCharts configuration in JSON format.
  
		  Examples:
		  1. Data: "firstName, lastName, sales, department\nJohn, Doe, 50000, Sales\nJane, Smith, 75000, Sales\nBob, Johnson, 30000, Marketing"
			 Current Config: { "options": { "chart": { "type": "bar" }, "xaxis": { "categories": ["Sales", "Marketing"] }, "yaxis": { "title": { "text": "Sales" } } }, "series": [{ "name": "Sales", "data": [125000, 30000] }] }
			 Prompt: "change to a pie chart"
			 Response: {
			   "options": { "chart": { "type": "pie" }, "labels": ["Sales", "Marketing"] },
			   "series": [125000, 30000]
			 }
		  2. Data: "firstName, lastName, sales, region\nJohn, Doe, 50000, North\nJane, Smith, 75000, South\nBob, Johnson, 30000, North"
			 Current Config: { "options": { "chart": { "type": "pie" }, "labels": ["North", "South"] }, "series": [80000, 75000] }
			 Prompt: "show sales by region as a bar chart"
			 Response: {
			   "options": { "chart": { "type": "bar" }, "xaxis": { "categories": ["North", "South"] }, "yaxis": { "title": { "text": "Sales" } } },
			   "series": [{ "name": "Sales", "data": [80000, 75000] }]
			 }
  
		  Now, analyze the following data and update the chart configuration based on the user's prompt.
		`;

			const content = `
		  The following table data is provided for analysis:
  
		  ${formattedData}
  
		  Now, please update the ApexCharts configuration for the following prompt:
		`;

			const initialMessages = [{ role: 'user', content: systemPrompt }];
			const userMessage = { role: 'user', content: `${content}\n${prompt}` };

			// Use generateText (non-streaming) for Config Mode
			const response = await generateText({
				model: google('gemini-1.5-flash'),
				messages: [...initialMessages, userMessage],
			});

			const resultText = response.text || response.data; // Adjust based on your API response structure
			console.log('Raw response:', resultText);

			const cleanedText = resultText
				.replace(/```json/g, '')
				.replace(/```/g, '')
				.trim();

			let updatedChartConfig;
			try {
				updatedChartConfig = JSON.parse(cleanedText);
			} catch (parseError) {
				console.error('Failed to parse chart config:', parseError);
				return c.text('Invalid chart configuration from AI', 500);
			}

			if (!updatedChartConfig.options || !updatedChartConfig.series) {
				return c.text('Invalid chart configuration: missing options or series', 400);
			}

			return c.json({ updatedChartConfig });
		} else {
		}
	} catch (error) {
		console.error('Error in /mrt-chart:', error);
		return c.text('Failed to process the request', 500);
	}
});

app.post('/generate-dashboard', async (c) => {
	const sampleData = {
		salesData: {
			monthly: [
				{ month: 'Jan', revenue: 12500, cost: 7000, profit: 5500 },
				{ month: 'Feb', revenue: 14200, cost: 7800, profit: 6400 },
				{ month: 'Mar', revenue: 15800, cost: 8200, profit: 7600 },
				{ month: 'Apr', revenue: 16900, cost: 8800, profit: 8100 },
				{ month: 'May', revenue: 19200, cost: 9500, profit: 9700 },
				{ month: 'Jun', revenue: 21500, cost: 10200, profit: 11300 },
			],
			products: [
				{ id: 1, name: 'Product A', sales: 1245, revenue: 62250 },
				{ id: 2, name: 'Product B', sales: 968, revenue: 48400 },
				{ id: 3, name: 'Product C', sales: 1492, revenue: 74600 },
				{ id: 4, name: 'Product D', sales: 856, revenue: 42800 },
				{ id: 5, name: 'Product E', sales: 1057, revenue: 52850 },
			],
			regions: [
				{ name: 'North', revenue: 45200 },
				{ name: 'South', revenue: 38900 },
				{ name: 'East', revenue: 42700 },
				{ name: 'West', revenue: 52600 },
				{ name: 'Central', revenue: 35800 },
			],
			kpis: {
				totalRevenue: 215200,
				totalProfit: 98600,
				averageOrderValue: 124.5,
				customerCount: 1728,
				revenueGrowth: 15.8,
				profitMargin: 45.8,
			},
		},

		userActivityData: {
			daily: [
				{ date: '2023-06-01', activeUsers: 2540, newUsers: 120, sessions: 4250 },
				{ date: '2023-06-02', activeUsers: 2620, newUsers: 145, sessions: 4380 },
				{ date: '2023-06-03', activeUsers: 2480, newUsers: 115, sessions: 4150 },
				{ date: '2023-06-04', activeUsers: 2390, newUsers: 98, sessions: 3980 },
				{ date: '2023-06-05', activeUsers: 2710, newUsers: 152, sessions: 4520 },
				{ date: '2023-06-06', activeUsers: 2850, newUsers: 168, sessions: 4760 },
				{ date: '2023-06-07', activeUsers: 2920, newUsers: 175, sessions: 4870 },
			],
			demographics: [
				{ age: '18-24', percentage: 22 },
				{ age: '25-34', percentage: 38 },
				{ age: '35-44', percentage: 25 },
				{ age: '45-54', percentage: 10 },
				{ age: '55+', percentage: 5 },
			],
			devices: [
				{ device: 'Mobile', sessions: 12580 },
				{ device: 'Desktop', sessions: 10340 },
				{ device: 'Tablet', sessions: 2480 },
			],
			kpis: {
				totalUsers: 8450,
				averageSessionDuration: '3m 24s',
				bounceRate: 32.5,
				conversionRate: 4.8,
				retentionRate: 68.2,
			},
		},
	};

	try {
		const body = await c.req.json();
		console.log('body', body);

		const { prompt } = body;

		// Validate input
		// if (!prompt || !data || !Array.isArray(data)) {
		// 	return c.text('Prompt and data are required', 400);
		// }

		// Format the data as a CSV-like string for better readability by the model
		// let formattedData = '';
		// if (Array.isArray(data) && data.length > 0) {
		// 	const headers = Object.keys(data[0]);
		// 	formattedData = headers.join(', ') + '\n';
		// 	formattedData += data.map((row) => headers.map((header) => row[header] ?? '').join(', ')).join('\n');
		// } else {
		// 	formattedData = 'No data available.';
		// }

		// Determine the mode based on whether config is provided
		// const isConfigMode = !!config;

		// Config Mode: Update the chart configuration (non-streaming)
		const llmPrompt = `
			You are a dashboard design expert. Based on the following prompt and data, create a JSON configuration for a dynamic dashboard with multiple components.
			
			User Prompt: "${prompt}"
			
			Available Data: ${JSON.stringify(sampleData, null, 2)}
			
			Respond with ONLY a valid JSON configuration for a dashboard with multiple components. The configuration should follow this format:
			
			{
			"title": "Dashboard Title",
			"description": "Dashboard description",
			"layout": "grid" or "tabbed",
			
			// For grid layout
			"components": [
				{
				"type": "chart",
				"gridSize": 6, // Size in MUI grid (1-12, where 12 is full width)
				"chartConfig": {
					"type": "bar|line|pie|area|radar|scatter|heatmap",
					"title": "Chart Title",
					"description": "Brief description",
					"height": 350,
					"data": {
					"categories": ["Category1", "Category2", ...],
					"series": [
						{
						"name": "Series Name",
						"data": [value1, value2, ...]
						}
					]
					},
					"xAxisTitle": "X-Axis Title",
					"yAxisTitle": "Y-Axis Title",
					"options": {}
				}
				},
				{
				"type": "table",
				"gridSize": 6,
				"tableConfig": {
					"title": "Table Title",
					"description": "Brief description",
					"dense": true,
					"columns": [
					{
						"header": "Column Name",
						"field": "fieldName",
						"align": "left|right|center"
					}
					],
					"data": [
					{
						"fieldName": "value",
						...
					}
					]
				}
				},
				{
				"type": "stat",
				"gridSize": 3,
				"statConfig": {
					"title": "Stat Title",
					"value": "Value",
					"change": 12.5, // Percentage change
					"period": "vs last period"
				}
				},
				{
				"type": "list",
				"gridSize": 6,
				"listConfig": {
					"title": "List Title",
					"description": "Brief description",
					"items": [
					{
						"primary": "Primary Text",
						"secondary": "Secondary Text",
						"avatar": "avatar_url",
						"value": "Value",
						"valueColor": "success.main",
						"highlighted": true
					}
					]
				}
				},
				{
				"type": "filter",
				"gridSize": 12,
				"filterConfig": {
					"title": "Filter Title",
					"type": "dropdown|toggle|tabs",
					"label": "Filter Label",
					"defaultValue": "value",
					"options": [
					{
						"label": "Option Label",
						"value": "option_value"
					}
					]
				}
				},
				{
				"type": "text",
				"gridSize": 12,
				"textConfig": {
					"title": "Text Title",
					"content": "Text content goes here",
					"variant": "body1|h5|subtitle1"
				}
				}
			],
			
			// For tabbed layout
			"tabs": [
				{
				"label": "Tab Label",
				"components": [
					// Same component objects as above
				]
				}
			]
			}
			
			Analyze the data and user's intent to create the most appropriate dashboard. Include a mix of different component types based on the prompt and available data.
		`;

		const initialMessages = [{ role: 'user', content: llmPrompt }];
		const userMessage = { role: 'user', content: `${prompt}` };

		// Use generateText (non-streaming) for Config Mode
		const response = await generateText({
			model: google('gemini-1.5-flash'),
			messages: [...initialMessages, userMessage],
		});

		const resultText = response.text || response.data; // Adjust based on your API response structure
		console.log('Raw response:', resultText);
		let dashboardConfig = null;
		// Extract JSON from the response text
		const jsonMatch = resultText.match(/\{[\s\S]*\}/);
		if (jsonMatch) {
			dashboardConfig = JSON.parse(jsonMatch[0]);
		} else {
			// Fallback if we can't extract JSON directly
			dashboardConfig = JSON.parse(resultText);
		}

		// const cleanedText = resultText
		// 	.replace(/```json/g, '')
		// 	.replace(/```/g, '')
		// 	.trim();

		return c.json({ success: true, data: sampleData, dashboardConfig, prompt });
	} catch (error) {
		console.error('Error in /mrt-chart:', error);
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
