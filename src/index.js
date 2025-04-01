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
const google = createGoogleGenerativeAI({
	apiKey: '',
});

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

app.post('/api/ai-chat', async (c) => {
	try {
		// Parse JSON request body
		const { query, context } = await c.req.json();

		if (!query) {
			console.warn('⚠️ Missing query in request');
			return c.text('Query is required', 400);
		}

		console.log('✅ Received query:', query);
		console.log('📊 Context data:', context);

		const systemPrompt2 = `You are an AI assistant called SuperPumped, specializing in Industrial IoT (IIoT) data analysis. 
		Your role is to analyze industrial data, detect patterns, and provide insights based on the given context. 
		Your responses should be precise, actionable, and data-driven.

		Use the following context for analysis:
		\n\n${JSON.stringify(context, null, 2)}

		`;

		// Initial AI conversation messages
		const initialMessages = [
			{ role: 'user', content: systemPrompt2 },
			{ role: 'assistant', content: 'Hello, how can I help?' },
		];

		const userMessage = { role: 'user', content: query };

		// Check if AI model is available
		// if (!aiModel) {
		// 	console.error('❌ AI model is not initialized!');
		// 	return c.text('AI model is unavailable. Try again later.', 500);
		// }

		// AI response
		const response = await streamText({
			model: google('gemini-2.0-flash-001'), // Use pre-initialized model
			messages: [...initialMessages, userMessage],
		});

		// Convert response to text stream
		return response.toTextStreamResponse();
	} catch (error) {
		console.error('🚨 Error in /api/chat:', error.message || error);
		return c.text('Failed to process the request. Please try again later.', 500);
	}
});

app.get('/api/chat', async (c) => {
	try {
		const query = c.req.query('query');
		console.log('query', query);

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
			model: google('gemini-2.5-pro-exp-03-25'),
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
		- If a column name in the prompt doesn't match the data, try to infer the closest match
		- Handle numerical comparisons (e.g., "above", "below") and aggregations (e.g., "average", "total") accurately.

		Now, analyze the following data and answer the user's prompt.
	  `;
		//   const systemPrompt = `
		//   You are an AI assistant called Superpumped that answers conversational queries about table data for a Material React Table (MRT) V2 component.
		//   The table data is provided in a CSV-like format, where the first row contains the column headers, and each subsequent row represents a data entry.
		//   Your task is to interpret the user's prompt and provide a concise, natural language response based on the data.

		//   Key Instructions:
		//   - The data is formatted as: "header1, header2, header3\nvalue1, value2, value3\n..."
		//   - Analyze the data to answer queries about highest/lowest values, averages, counts, filtering, or other aggregations.
		//   - Respond in a conversational tone, as if speaking to the user directly.
		//   - Do not modify the data or table configuration; only provide a textual response.
		//   - If a column name in the prompt doesn't match the data, try to infer the closest match (e.g., "sales" might be "Sales").
		//   - Handle numerical comparisons (e.g., "above", "below") and aggregations (e.g., "average", "total") accurately.
		//   - For queries involving names, use both firstName and lastName if available (e.g., "John Doe").

		//   Examples:
		//   1. Data: "firstName, lastName, sales\nJohn, Doe, 50000\nJane, Smith, 75000"
		// 	 Prompt: "who has the highest sales?"
		// 	 Response: "Jane Smith has the highest sales with a value of 75000."
		//   2. Data: "firstName, lastName, salary, department\nJohn, Doe, 60000, Sales\nJane, Smith, 80000, Sales\nBob, Johnson, 55000, Marketing"
		// 	 Prompt: "average salary in each department"
		// 	 Response: "Average salary in Sales: 70000. Average salary in Marketing: 55000."
		//   3. Data: "firstName, lastName, region\nJohn, Doe, North\nJane, Smith, South\nBob, Johnson, North"
		// 	 Prompt: "how many people in each region?"
		// 	 Response: "There are 2 people in North and 1 person in South."
		//   4. Data: "firstName, lastName, salary, department\nJohn, Doe, 60000, Sales\nJane, Smith, 80000, Sales"
		// 	 Prompt: "list people in Sales with salary above 70000"
		// 	 Response: "Jane Smith has a salary of 80000 in Sales."
		//   5. Data: "firstName, lastName, age\nJohn, Doe, 30\nJane, Smith, 25"
		// 	 Prompt: "who is the youngest?"
		// 	 Response: "Jane Smith is the youngest with an age of 25."

		//   Now, analyze the following data and answer the user's prompt.
		// `;
		console.log('formattedData', formattedData);
		const content = `
		The following table data is provided for analysis:
		${formattedData}  
		Now, please answer the following question:
	  `;

		const initialMessages = [{ role: 'user', content: systemPrompt }];
		const userMessage = { role: 'user', content: `${content}\n${prompt}` };

		const response = await streamText({
			model: google('gemini-2.5-pro-exp-03-25'),
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
			// 	const systemPrompt = `
			//   You are an AI assistant called Superpumped that generates and updates chart configurations for ApexCharts based on table data and user prompts.
			//   The table data is provided in a CSV-like format, where the first row contains the column headers, and each subsequent row represents a data entry.
			//   The current chart configuration is provided in JSON format.
			//   Your task is to interpret the user's prompt and return a JSON object representing the updated ApexCharts configuration, including:
			//   - "options": The chart options (e.g., chart type, xaxis, yaxis, labels).
			//   - "series": The data series for the chart (e.g., [{ name: "Sales", data: [50000, 75000] }]).

			//   Key Instructions:
			//   - The data is formatted as: "header1, header2, header3\nvalue1, value2, value3\n..."
			//   - The current chart config is: ${JSON.stringify(config)}.
			//   - Determine the appropriate chart type based on the prompt (e.g., "bar chart" → bar, "pie chart" → pie, "line chart" → line).
			//   - If the chart type is not specified, keep the current chart type or default to a bar chart for numerical data grouped by a categorical column.
			//   - Aggregate data as needed (e.g., sum, average, count) based on the prompt.
			//   - For bar charts, use a categorical column (e.g., department) for the x-axis and a numerical column (e.g., sales) for the y-axis.
			//   - For pie charts, use a categorical column for labels and a numerical column for values.
			//   - For line charts, ensure there’s a time-based or sequential column for the x-axis.
			//   - If the prompt is unclear or the data doesn’t support the requested chart, return the current chart config with a message in the response.
			//   - Preserve existing chart options unless the prompt explicitly changes them.
			//   - Return a valid ApexCharts configuration in JSON format.

			//   Examples:
			//   1. Data: "firstName, lastName, sales, department\nJohn, Doe, 50000, Sales\nJane, Smith, 75000, Sales\nBob, Johnson, 30000, Marketing"
			// 	 Current Config: { "options": { "chart": { "type": "bar" }, "xaxis": { "categories": ["Sales", "Marketing"] }, "yaxis": { "title": { "text": "Sales" } } }, "series": [{ "name": "Sales", "data": [125000, 30000] }] }
			// 	 Prompt: "change to a pie chart"
			// 	 Response: {
			// 	   "options": { "chart": { "type": "pie" }, "labels": ["Sales", "Marketing"] },
			// 	   "series": [125000, 30000]
			// 	 }
			//   2. Data: "firstName, lastName, sales, region\nJohn, Doe, 50000, North\nJane, Smith, 75000, South\nBob, Johnson, 30000, North"
			// 	 Current Config: { "options": { "chart": { "type": "pie" }, "labels": ["North", "South"] }, "series": [80000, 75000] }
			// 	 Prompt: "show sales by region as a bar chart"
			// 	 Response: {
			// 	   "options": { "chart": { "type": "bar" }, "xaxis": { "categories": ["North", "South"] }, "yaxis": { "title": { "text": "Sales" } } },
			// 	   "series": [{ "name": "Sales", "data": [80000, 75000] }]
			// 	 }

			//   Now, analyze the following data and update the chart configuration based on the user's prompt.
			// `;
			const systemPrompt = `
			You are ChartGPT, an AI assistant specialized in generating optimal ApexCharts configurations based on data analysis and user requests.
			
			DATA FORMAT:
			${formattedData}
			
			CURRENT CONFIGURATION:
			${JSON.stringify(config, null, 2)}
			
			Your task is to analyze the data and user request, then provide a precise ApexCharts configuration JSON that best visualizes the insights.
			
			CRITICAL: Your response MUST be valid, parseable JSON. DO NOT include JavaScript functions in your JSON as they will cause parsing errors.
			
		FORMATTER PLACEHOLDERS:
			Instead of including function definitions directly, use ONLY these specific placeholder strings:

			1. General value formatters:
			- "VALUE_FORMATTER" - Displays values with 0 decimal places
			- "CURRENCY_FORMATTER" - Adds $ and displays with 0 decimal places
			- "PERCENTAGE_FORMATTER" - Adds % and displays with 1 decimal place
			- "K_FORMATTER" - Divides by 1000 and adds 'k' suffix
			- "SHORT_NUMBER_FORMATTER" - Smart abbreviation (1.2M, 5.4k) based on value size

			2. Date formatters:
			- "DATE_FORMATTER" - Standard date format (MM/DD/YYYY)
			- "TIMESTAMP_FORMATTER" - Date with time
			- "MONTH_YEAR_FORMATTER" - Month and year only (Jan 2023)

			3. Special formatters:
			- "CUSTOM_LEGEND_FORMATTER" - For legend formatting
			- "DYNAMIC_TITLE_FORMATTER" - For dynamic chart titles

			Example of INCORRECT (will cause errors):
			{
			"tooltip": {
				"y": {
				"formatter": function(val) { return "$" + val }
				"title": {formatter: "(seriesName) => seriesName + ': '"}
				},				
			}
			}

			Example of CORRECT (will parse successfully):
			{
			"tooltip": {
				"y": {
				"formatter": "CURRENCY_FORMATTER"
				"title":{
					"formatter":"SERIES_FORMATTER"
					}
				}
			}
			}
			*** Neve ever use function expression as formatter in any case ***
			RESPONSE FORMAT:
			Return ONLY a valid JSON object with two main properties:
			- "options": Chart display settings (type, axes, colors, title, etc.)
			- "series": Data series formatted appropriately for the chart type
			
			CHART TYPE SELECTION GUIDELINES:
			- Bar charts: Best for comparing quantities across categories
			- Line charts: Ideal for showing trends over time or sequences
			- Pie/Donut charts: Effective for showing proportions of a whole (limit to 7 categories max)
			- Area charts: Good for showing cumulative totals over time
			- Column charts: Similar to bar but vertical, good for time-based comparisons
			- Scatter plots: Best for showing correlation between two variables
			- Heatmaps: Excellent for showing patterns in complex datasets
			
			DATA PROCESSING RULES:
			1. Automatically aggregate data when appropriate (sum, average, count)
			2. For time series, sort chronologically before plotting
			3. Limit displayed categories to improve readability (max 10-12 items)
			4. Format numbers appropriately using string placeholders instead of functions
			5. Choose appropriate scales to highlight patterns without distortion
			
			DESIGN BEST PRACTICES:
			1. Use clear, descriptive titles and axis labels
			2. Select a color palette appropriate for the data (sequential, diverging, categorical)
			3. Add tooltips with detailed information for interactive exploration
			4. Include legends when multiple series are present
			5. Set appropriate grid lines and tick marks for readability
			6. Format dates in user-friendly ways (Month Year instead of timestamps)
			7. For dense datasets, consider adding zoom/pan capabilities
			
			CUSTOMIZATION PRIORITIES:
			1. If user specifies a chart type, use it unless fundamentally inappropriate
			2. Respect explicit user preferences for colors, labels, and formatting
			3. Preserve existing configuration elements unless explicitly changed
			4. If the user request conflicts with best practices, prioritize clarity and accuracy
			
			ERROR HANDLING:
			If the data cannot support the requested visualization, explain why in a comment property and provide the best alternative configuration.
			
			Analyze the data structure carefully before configuring. Identify numerical columns for measures and categorical/date columns for dimensions.
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
				model: google('gemini-2.5-pro-exp-03-25'),
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

app.post('/process-file', async (c) => {
	const formData = await c.req.formData();
	const file = formData.get('file');

	// Validate inputs
	if (!file) {
		return c.json({ error: 'Missing file' }, 400);
	}
	const prompt = formData.get('prompt') || '';
	if (!prompt) {
	}

	console.log('file', file);
	console.log('prompt', prompt);
	try {
		// Read file based on type
		const arrayBuffer = await file.arrayBuffer();
		let rawData;

		if (file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') {
			// Excel (.xlsx)
			const workbook = read(arrayBuffer, { type: 'array' });
			const sheetName = workbook.SheetNames[0];
			const worksheet = workbook.Sheets[sheetName];
			rawData = utils.sheet_to_json(worksheet);
		} else if (file.type === 'text/csv') {
			// CSV
			const text = new TextDecoder().decode(arrayBuffer);
			rawData = parseCSV(text); // Custom CSV parsing
		} else {
			return c.json({ error: 'Unsupported file type' }, 400);
		}
		console.log('here 1');
		if (!prompt) {
			return c.json({ data: rawData });
		}
		console.log('here 2');
		// Process data with AI
		const processedData = await processDataWithAI(rawData, prompt);
		return c.json({
			data: processedData,
			// dashboardConfig: generateDashboardConfig(processedData),
		});
	} catch (error) {
		return c.json({ error: `Failed to process file: ${error.message}` }, 500);
	}
});

function parseCSV(text) {
	const lines = text.split('\n').filter((line) => line.trim() !== '');
	const headers = lines[0].split(',').map((h) => h.trim());
	const data = lines.slice(1).map((line) => {
		const values = line.split(',').map((v) => v.trim());
		return headers.reduce((obj, header, i) => {
			obj[header] = isNaN(values[i]) ? values[i] : Number(values[i]);
			return obj;
		}, {});
	});
	return data;
}

async function processDataWithAI(rawData, prompt) {
	// const apiKey = 'YOUR_GOOGLE_AI_API_KEY'; // Store in environment variables
	// const apiUrl = 'https://api.google-ai-endpoint.com/v1/models/gemini:generate'; // Adjust URL

	const aiPrompt = `
	  Given the following dataset: 
	  ${JSON.stringify(rawData, null, 2)}
	  Perform the following operations based on this user prompt: "${prompt}"
	  Return the transformed data in JSON format.
	  ONLY RETURN THE DATA AND NOT ANY EXPLAINATION AT ALL.
	`;

	const initialMessages = [{ role: 'user', content: aiPrompt }];
	// const userMessage = { role: 'user', content: `${prompt}` };

	// Use generateText (non-streaming) for Config Mode
	const response = await generateText({
		model: google('gemini-2.0-flash-001'),
		messages: [...initialMessages],
	});

	const resultText = response.text || response.data; // Adjust based on your API response structure
	console.log('Raw response:', resultText);

	// if (!response.ok) {
	// 	throw new Error('AI processing failed');
	// }

	const cleanedText = resultText
		.replace(/```json/g, '') // Remove opening ```json
		.replace(/```/g, '') // Remove closing ```
		.trim(); // Remove leading/trailing whitespace

	console.log('cleanedText', cleanedText);
	return JSON.parse(cleanedText);
	// Try to extract valid JSON using a more specific regular expression
	// const jsonMatch = cleanedResponse.match(/\{[\s\S]*\}/);

	// if (!jsonMatch) {
	// 	throw new Error('No valid JSON found in the response');
	// }

	// const parsedJson = JSON.parse(jsonMatch[0]); // Parse the extracted JSON string

	// console.log('Parsed JSON:', parsedJson);

	// return parsedJson;
	// const result = await jsonMatch.json();
	// return JSON.parse(result.generatedText); // Assuming AI returns JSON string
}

app.post('/generate-dashboard', async (c) => {
	const sampleData = {
		data: [
			{
				Line: 'Line 1 (Auto- Uni)',
				'SKU Information': 'RSWNSW2UTPCVL',
				'Final FG': 4718,
				'Sensor Final FG': 4306,
				'Opening Hours (min)': 915,
				'Weighted Line Rating': 7624.58,
				'Weighted Line Rating (Constraint)': 7624.58,
				'Machine Losses (min)': 0,
				'Utility Losses (min)': 0,
				'Major Stoppages': 56,
				'Minor Stoppages': 342,
				'Unbooked Major Downtime': 0,
				'Unbooked Minor Downtime': 6782,
				'Planned Downtime (min)': 0,
				'Changeover Loss (min)': 0,
				'DG/WG Unavailability (min)': 0,
				'Quality Loss / Checks (min)': 93,
				'Cleaning (min)': 0,
				'Non Operational time (min)': 0,
				'Line Start Up and Shut Down Loss (min)': 5,
				'Common/Other Losses (min)': 1,
				'Performance Losses (min)': 169,
				'Machine Unavailability Losses (min)': 0,
				'Operating Hours (min)': 915,
				'Machine Hours (min)': 613,
				'Factory Efficiency': 56.48,
				'Operating Efficiency': 56.48,
				'Rated Performance Rate': 92.4,
				'Sensor Rated Performance Rate': 84.1,
				'Constraint Performance Rate': 84.1,
				'Machine Reliability': 91.72,
				Availability: 67,
				OEE: 61.9,
				'Sensor OEE': 56.35,
				'Constraint OEE': 56.35,
			},
			{
				Line: 'Line 1 (Auto- Uni)',
				'SKU Information': 'RSWNSW2UTPCVL-RSWNSW2UTPCVL',
				'Final FG': 2104,
				'Sensor Final FG': 2104,
				'Opening Hours (min)': 709,
				'Weighted Line Rating': 5187.22,
				'Weighted Line Rating (Constraint)': 5187.22,
				'Machine Losses (min)': 0,
				'Utility Losses (min)': 0,
				'Major Stoppages': 72,
				'Minor Stoppages': 155,
				'Unbooked Major Downtime': 5436,
				'Unbooked Minor Downtime': 3273,
				'Planned Downtime (min)': 0,
				'Changeover Loss (min)': 5,
				'DG/WG Unavailability (min)': 0,
				'Quality Loss / Checks (min)': 110,
				'Cleaning (min)': 0,
				'Non Operational time (min)': 0,
				'Line Start Up and Shut Down Loss (min)': 0,
				'Common/Other Losses (min)': 10,
				'Performance Losses (min)': 0,
				'Machine Unavailability Losses (min)': 0,
				'Operating Hours (min)': 709,
				'Machine Hours (min)': 377,
				'Factory Efficiency': 40.56,
				'Operating Efficiency': 40.56,
				'Rated Performance Rate': 67,
				'Sensor Rated Performance Rate': 67,
				'Constraint Performance Rate': 67,
				'Machine Reliability': 97.86,
				Availability: 53.2,
				OEE: 35.62,
				'Sensor OEE': 35.62,
				'Constraint OEE': 35.62,
			},
			{
				Line: 'Line 2 (Auto- Uni)',
				'SKU Information': 'IBWNDC1UTPCVL',
				'Final FG': 5483,
				'Sensor Final FG': 5007,
				'Opening Hours (min)': 915,
				'Weighted Line Rating': 4765.02,
				'Weighted Line Rating (Constraint)': 4765.02,
				'Machine Losses (min)': 0,
				'Utility Losses (min)': 0,
				'Major Stoppages': 41,
				'Minor Stoppages': 337,
				'Unbooked Major Downtime': 0,
				'Unbooked Minor Downtime': 6939,
				'Planned Downtime (min)': 0,
				'Changeover Loss (min)': 0,
				'DG/WG Unavailability (min)': 0,
				'Quality Loss / Checks (min)': 2,
				'Cleaning (min)': 0,
				'Non Operational time (min)': 0,
				'Line Start Up and Shut Down Loss (min)': 8,
				'Common/Other Losses (min)': 0,
				'Performance Losses (min)': 65,
				'Machine Unavailability Losses (min)': 0,
				'Operating Hours (min)': 915,
				'Machine Hours (min)': 659,
				'Factory Efficiency': 105.08,
				'Operating Efficiency': 105.08,
				'Rated Performance Rate': 159.7,
				'Sensor Rated Performance Rate': 146.1,
				'Constraint Performance Rate': 146.1,
				'Machine Reliability': 80.22,
				Availability: 72.1,
				OEE: 115.11,
				'Sensor OEE': 105.31,
				'Constraint OEE': 105.32,
			},
		],
	};

	try {
		const body = await c.req.json();
		console.log('body', body);

		const { prompt, data } = body;

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
		// const llmPrompt = `
		// 	You are a dashboard design expert. Based on the following prompt and data, create a JSON configuration for a dynamic dashboard with multiple components.

		// 	User Prompt: "${prompt}"

		// 	Available Data: ${JSON.stringify(sampleData, null, 2)}

		// 	Respond with ONLY a valid JSON configuration for a dashboard with multiple components. The configuration should follow this format:

		// 	{
		// 	"title": "Dashboard Title",
		// 	"description": "Dashboard description",
		// 	"layout": "grid" or "tabbed",

		// 	// For grid layout
		// 	"components": [
		// 		{
		// 		"type": "chart",
		// 		"gridSize": 6, // Size in MUI grid (1-12, where 12 is full width)
		// 		"chartConfig": {
		// 			"type": "bar|line|pie|area|radar|scatter|heatmap",
		// 			"title": "Chart Title",
		// 			"description": "Brief description",
		// 			"height": 350,
		// 			"data": {
		// 			"categories": ["Category1", "Category2", ...],
		// 			"series": [
		// 				{
		// 				"name": "Series Name",
		// 				"data": [value1, value2, ...]
		// 				}
		// 			]
		// 			},
		// 			"xAxisTitle": "X-Axis Title",
		// 			"yAxisTitle": "Y-Axis Title",
		// 			"options": {}
		// 		}
		// 		},
		// 		{
		// 		"type": "table",
		// 		"gridSize": 6,
		// 		"tableConfig": {
		// 			"title": "Table Title",
		// 			"description": "Brief description",
		// 			"dense": true,
		// 			"columns": [
		// 			{
		// 				"header": "Column Name",
		// 				"field": "fieldName",
		// 				"align": "left|right|center"
		// 			}
		// 			],
		// 			"data": [
		// 			{
		// 				"fieldName": "value",
		// 				...
		// 			}
		// 			]
		// 		}
		// 		},
		// 		{
		// 		"type": "stat",
		// 		"gridSize": 3,
		// 		"statConfig": {
		// 			"title": "Stat Title",
		// 			"value": "Value",
		// 			"change": 12.5, // Percentage change
		// 			"period": "vs last period"
		// 		}
		// 		},
		// 		{
		// 		"type": "list",
		// 		"gridSize": 6,
		// 		"listConfig": {
		// 			"title": "List Title",
		// 			"description": "Brief description",
		// 			"items": [
		// 			{
		// 				"primary": "Primary Text",
		// 				"secondary": "Secondary Text",
		// 				"avatar": "avatar_url",
		// 				"value": "Value",
		// 				"valueColor": "success.main",
		// 				"highlighted": true
		// 			}
		// 			]
		// 		}
		// 		},
		// 		{
		// 		"type": "filter",
		// 		"gridSize": 12,
		// 		"filterConfig": {
		// 			"title": "Filter Title",
		// 			"type": "dropdown|toggle|tabs",
		// 			"label": "Filter Label",
		// 			"defaultValue": "value",
		// 			"options": [
		// 			{
		// 				"label": "Option Label",
		// 				"value": "option_value"
		// 			}
		// 			]
		// 		}
		// 		},
		// 		{
		// 		"type": "text",
		// 		"gridSize": 12,
		// 		"textConfig": {
		// 			"title": "Text Title",
		// 			"content": "Text content goes here",
		// 			"variant": "body1|h5|subtitle1"
		// 		}
		// 		}
		// 	],

		// 	// For tabbed layout
		// 	"tabs": [
		// 		{
		// 		"label": "Tab Label",
		// 		"components": [
		// 			// Same component objects as above
		// 		]
		// 		}
		// 	]
		// 	}

		// 	Analyze the data and user's intent to create the most appropriate dashboard. Include a mix of different component types based on the prompt and available data.
		// `;

		const llmPrompt = `
	You are an expert in dashboard design and data visualization, specializing in ApexCharts. Based on the user's prompt and the provided raw data, generate a JSON configuration for a dynamic dashboard with multiple components (charts, tables, stats, etc.) that delivers precise, error-free insights tailored to the user's intent. The dashboard must use ApexCharts for all chart components, and the configuration must match ApexCharts' expected structure.

	User Prompt: "${prompt}"

	Available Data: ${JSON.stringify(data, null, 2)}

	Your task:
	1. Analyze the raw data and the user's prompt to determine the most effective way to process it for maximum insight and accuracy. Options include:
		- Grouping or aggregating data by a field (e.g., a categorical column like "department" or "Line") if it reveals trends or comparisons.
		- Calculating sums, averages, or other statistics for numeric fields based on their semantic meaning (e.g., totals for counts, averages for percentages).
		- Preserving individual records if detailed granularity is more valuable.
		- Identifying and highlighting trends, outliers, or performance gaps.
	2. Special handling for duplicate fields:
		- If a field (e.g., "Line") has multiple entries with different values in another field (e.g., "SKU Information"), treat each entry as distinct rather than aggregating or skipping them.
		- When confusion arises (e.g., in a "Line vs OEE" chart), list all relevant entries separately and include the differentiating factor (e.g., "SKU Information") in labels or descriptions for clarity.
	3. Explore the full dataset:
		- Analyze all available fields and prioritize those critical for decision-making (e.g., production output, efficiency metrics, downtime, etc.).
		- Don’t limit focus to a few fields—use the richness of the data to provide a comprehensive view.
	4. Design a dashboard with a diverse mix of components (charts, tables, stats) that:
		- Includes at least 3-5 components to cover different aspects of the data (e.g., comparisons, proportions, correlations, summaries).
		- Uses a variety of chart types appropriately matched to the data and prompt:
			- Bar charts for comparisons of numerical values across categories (e.g., "Line vs OEE", "sales by department"). Prefer bar charts for comparisons unless the prompt explicitly requests a different chart type.
			- Column charts (a variant of bar charts) for multi-series comparisons over categories.
			- Pie charts for proportions (e.g., breakdown of downtime categories), ensuring categories are meaningful, non-zero, and correctly labeled.
			- Line charts for trends (if a time-based or sequential column is present).
			- Area charts for trends with filled areas, especially for cumulative data.
			- Scatter charts for correlations (e.g., two numerical fields like "OEE vs downtime").
			- Radar charts for comparing multiple variables across categories.
			- Heatmap charts for visualizing data intensity across two dimensions (e.g., time vs category).
		- Visualizes key metrics with precision and clarity across multiple dimensions.
		- Enables comparisons across relevant categories, ensuring all distinct entries are represented.
		- Highlights actionable insights (e.g., inefficiencies, bottlenecks, top performers).
		- Adapts dynamically to the data structure and prompt without assumptions.
	5. Ensure chart data integrity:
		- For all charts, only include non-zero values and ensure categories/labels are explicitly labeled with the corresponding field names (e.g., "Unbooked Minor Downtime" instead of "series-1").
		- For pie charts, map categorical labels to field names and numerical values to non-zero data points.
		- If a chart cannot be meaningfully generated (e.g., no non-zero data for a pie chart, or no time data for a line chart), replace it with a more suitable component like a stat or table summarizing the data.
	6. Handle dynamic formatters in JSON:
		- Do NOT include JavaScript functions directly in the JSON output (e.g., "formatter": function() {...}).
		- Include stringified functions with correct ApexCharts parameter names:
		- Use "function(val) { return val + ' %'; }" for data labels (e.g., percentage formatting).
		- Use "function(val, { seriesIndex, dataPointIndex, w }) { return w.globals.labels[dataPointIndex] + ': ' + val + ' %'; }" for tooltips.
		- Only include formatters when explicitly needed (e.g., for dataLabels or tooltip), otherwise omit them
	7. Ensure the output is error-free, avoids generic placeholders (e.g., "series-1"), and directly reflects the provided data.

	Respond with ONLY a valid JSON configuration for a dashboard with multiple components, following this format:

	{
		"title": "Dashboard Title",
		"description": "Dashboard description",
		"layout": "grid",
		"components": [
			{
				"type": "chart",
				"gridSize": 6,
				"chartConfig": {
					// Line Chart Example
					"type": "line",
					"title": "Product Trends by Month",
					"description": "Line chart showing product trends over months",
					"height": 350,
					"series": [
						{
							"name": "Desktops",
							"data": [10, 41, 35, 51, 49, 62, 69, 91, 148]
						}
					],
					"options": {
						"chart": {
							"height": 350,
							"type": "line",
							"zoom": {
								"enabled": false
							}
						},
						"dataLabels": {
							"enabled": false
						},
						"stroke": {
							"curve": "straight"
						},
						"title": {
							"text": "Product Trends by Month",
							"align": "left"
						},
						"grid": {
							"row": {
								"colors": ["#f3f3f3", "transparent"],
								"opacity": 0.5
							}
						},
						"xaxis": {
							"categories": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
						}
					}
				}
			},
			{
				"type": "chart",
				"gridSize": 6,
				"chartConfig": {
					// Bar Chart Example (Horizontal)
					"type": "bar",
					"title": "Sales by Country",
					"description": "Horizontal bar chart showing sales by country",
					"height": 350,
					"series": [
						{
							"data": [400, 430, 448, 470, 540, 580, 690, 1100, 1200, 1380]
						}
					],
					"options": {
						"chart": {
							"height": 350,
							"type": "bar"
						},
						"plotOptions": {
							"bar": {
								"borderRadius": 4,
								"borderRadiusApplication": "end",
								"horizontal": true
							}
						},
						"dataLabels": {
							"enabled": false
						},
						"xaxis": {
							"categories": ["South Korea", "Canada", "United Kingdom", "Netherlands", "Italy", "France", "Japan", "United States", "China", "Germany"]
						}
					}
				}
			},
			{
				"type": "chart",
				"gridSize": 6,
				"chartConfig": {
					// Column Chart Example (Vertical Bar with Multiple Series)
					"type": "bar",
					"title": "Financial Metrics by Month",
					"description": "Column chart showing financial metrics over months",
					"height": 350,
					"series": [
						{
							"name": "Net Profit",
							"data": [44, 55, 57, 56, 61, 58, 63, 60, 66]
						},
						{
							"name": "Revenue",
							"data": [76, 85, 101, 98, 87, 105, 91, 114, 94]
						},
						{
							"name": "Free Cash Flow",
							"data": [35, 41, 36, 26, 45, 48, 52, 53, 41]
						}
					],
					"options": {
						"chart": {
							"height": 350,
							"type": "bar"
						},
						"plotOptions": {
							"bar": {
								"horizontal": false,
								"columnWidth": "55%",
								"borderRadius": 5,
								"borderRadiusApplication": "end"
							}
						},
						"dataLabels": {
							"enabled": false
						},
						"stroke": {
							"show": true,
							"width": 2,
							"colors": ["transparent"]
						},
						"xaxis": {
							"categories": ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
						},
						"yaxis": {
							"title": {
								"text": "$ (thousands)"
							}
						},
						"fill": {
							"opacity": 1
						}
					}
				}
			},
			{
				"type": "chart",
				"gridSize": 6,
				"chartConfig": {
					// Pie Chart Example
					"type": "pie",
					"title": "Team Contributions",
					"description": "Pie chart showing team contributions",
					"height": 350,
					"series": [44, 55, 13, 43, 22],
					"options": {
						"chart": {
							"width": 380,
							"type": "pie"
						},
						"labels": ["Team A", "Team B", "Team C", "Team D", "Team E"],
						"dataLabels": {
							"enabled": true,
							"formatter": "percentageLabel"
						},
						"responsive": [
							{
								"breakpoint": 480,
								"options": {
									"chart": {
										"width": 200
									},
									"legend": {
										"position": "bottom"
									}
								}
							}
						]
					}
				}
			},
			{
				"type": "chart",
				"gridSize": 6,
				"chartConfig": {
					// Area Chart Example
					"type": "area",
					"title": "Stock Price Movements",
					"description": "Area chart showing stock price trends over time",
					"height": 350,
					"series": [
						{
							"name": "STOCK ABC",
							"data": [31, 40, 28, 51, 42, 109, 100]
						}
					],
					"options": {
						"chart": {
							"height": 350,
							"type": "area",
							"zoom": {
								"enabled": false
							}
						},
						"dataLabels": {
							"enabled": false
						},
						"stroke": {
							"curve": "straight"
						},
						"title": {
							"text": "Stock Price Movements",
							"align": "left"
						},
						"subtitle": {
							"text": "Price Movements",
							"align": "left"
						},
						"labels": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01", "2023-07-01"],
						"xaxis": {
							"type": "datetime"
						},
						"yaxis": {
							"opposite": true
						},
						"legend": {
							"horizontalAlign": "left"
						}
					}
				}
			},
			{
				"type": "chart",
				"gridSize": 6,
				"chartConfig": {
					// Scatter Chart Example
					"type": "scatter",
					"title": "Sample Data Scatter",
					"description": "Scatter chart showing sample data points",
					"height": 350,
					"series": [
						{
							"name": "SAMPLE A",
							"data": [
								[16.4, 5.4], [21.7, 2], [25.4, 3]
							]
						},
						{
							"name": "SAMPLE B",
							"data": [
								[36.4, 13.4], [1.7, 11], [5.4, 8]
							]
						}
					],
					"options": {
						"chart": {
							"height": 350,
							"type": "scatter",
							"zoom": {
								"enabled": true,
								"type": "xy"
							}
						},
						"xaxis": {
							"tickAmount": 10,
							"labels": {
								"formatter": "decimalLabel"
							}
						},
						"yaxis": {
							"tickAmount": 7
						}
					}
				}
			},
			{
				"type": "chart",
				"gridSize": 6,
				"chartConfig": {
					// Radar Chart Example
					"type": "radar",
					"title": "Performance Metrics by Category",
					"description": "Radar chart comparing performance metrics across categories",
					"height": 350,
					"series": [
						{
							"name": "Series 1",
							"data": [80, 50, 30, 40, 100, 20]
						},
						{
							"name": "Series 2",
							"data": [50, 60, 70, 20, 80, 90]
						}
					],
					"options": {
						"chart": {
							"height": 350,
							"type": "radar"
						},
						"dataLabels": {
							"enabled": true,
							"formatter": "valueLabel"
						},
						"xaxis": {
							"categories": ["Speed", "Power", "Efficiency", "Durability", "Accuracy", "Reliability"]
						},
						"yaxis": {
							"max": 100
						}
					}
				}
			},
			{
				"type": "chart",
				"gridSize": 6,
				"chartConfig": {
					// Heatmap Chart Example
					"type": "heatmap",
					"title": "Activity Heatmap by Day and Hour",
					"description": "Heatmap showing activity intensity across days and hours",
					"height": 350,
					"series": [
						{
							"name": "Mon",
							"data": [10, 20, 30, 40, 50, 60, 70]
						},
						{
							"name": "Tue",
							"data": [15, 25, 35, 45, 55, 65, 75]
						},
						{
							"name": "Wed",
							"data": [20, 30, 40, 50, 60, 70, 80]
						}
					],
					"options": {
						"chart": {
							"height": 350,
							"type": "heatmap"
						},
						"dataLabels": {
							"enabled": false
						},
						"xaxis": {
							"categories": ["9AM", "10AM", "11AM", "12PM", "1PM", "2PM", "3PM"]
						},
						"colors": ["#008FFB"]
					}
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
							"align": "left"
						}
					],
					"data": [
						{
							"fieldName": "value"
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
					"change": 12.5,
					"period": "vs last period"
				}
			}
		]
	}

	The dashboard must be precise, actionable, visually diverse, and directly tied to the input data and prompt—no dummy data, placeholders, generic labels, or JavaScript functions allowed. Ensure all chart configurations are valid for ApexCharts and match the structure shown in the examples. The "series" field must always be a top-level field in "chartConfig", separate from "options".
`;

		const llmPrompt2 = `
You are an expert in dashboard design and data visualization specializing in ApexCharts. Based on the user's prompt and the provided raw data, your task is to generate a fully valid JSON configuration for a dynamic dashboard with multiple components (charts, tables, stats, etc.) that delivers precise, actionable insights directly tied to the data. Your JSON must match ApexCharts' expected structure, using the provided examples as a guide, and must include all relevant components.

User Prompt: "${prompt}"

Available Data: ${JSON.stringify(data, null, 2)}

Instructions:
1. Analyze the provided raw data and the user's prompt to determine how to best aggregate, group, and process the data. Consider:
   - Grouping or aggregating data by relevant categorical fields (e.g., "department", "Line") to reveal trends.
   - Computing sums, averages, or other relevant statistics from numeric fields.
   - Preserving detailed individual records when granularity is critical.
   - Highlighting trends, outliers, and performance gaps.

2. When encountering duplicate fields (e.g., multiple "SKU Information" entries for the same "Line"):
   - Treat each as distinct, and include distinguishing information (like "SKU Information") in labels or descriptions.

3. Explore the entire dataset and prioritize fields that drive decision-making (e.g., production output, efficiency metrics, downtime).

4. Design a dashboard that includes at least 3-5 components. The components must be diverse and directly reflect the data:
   - Use bar charts for comparing numerical values across categories.
   - Use column charts for multi-series comparisons over categories.
   - Use pie charts for proportions (ensuring only non-zero values are included).
   - Use line charts for trends if there is a time-based or sequential field.
   - Use area charts when a filled trend visualization is more appropriate.
   - Use scatter charts for visualizing correlations between two numerical fields.
   - Use radar charts for comparing multiple performance metrics.
   - Use heatmap charts to show intensity across two dimensions (e.g., time vs category).
   - If a specific chart type cannot be generated meaningfully, substitute it with a table or stat component summarizing the data.

5. Ensure chart data integrity:
   - Only include non-zero values.
   - Use explicit, clear labels mapped directly from field names (e.g., "Unbooked Minor Downtime" instead of "series-1").
   - For pie charts, ensure categories are meaningful and paired with their non-zero values.
   - The "series" field must be a top-level property within "chartConfig", separate from "options".

6. Do not include any generic placeholders, dummy data, or JavaScript functions in the JSON. If a formatter is needed, include it as a string (e.g., "function(val) { return val + ' %'; }") following the correct ApexCharts parameters, but only when explicitly required.

7. The final JSON configuration must be error-free and structured as follows (each component must reflect real data from the provided dataset and prompt):

{
  "title": "Dashboard Title",
  "description": "Dashboard description",
  "layout": "grid",
  "components": [
    {
      "type": "chart",
      "gridSize": 6,
      "chartConfig": {
        "type": "line",
        "title": "Product Trends by Month",
        "description": "Line chart showing product trends over months",
        "height": 350,
        "series": [ ... ],
        "options": { ... }
      }
    },
    {
      "type": "chart",
      "gridSize": 6,
      "chartConfig": {
        "type": "bar",
        "title": "Sales by Country",
        "description": "Horizontal bar chart showing sales by country",
        "height": 350,
        "series": [ ... ],
        "options": { ... }
      }
    },
    {
      "type": "chart",
      "gridSize": 6,
      "chartConfig": {
        "type": "bar",
        "title": "Financial Metrics by Month",
        "description": "Column chart showing financial metrics over months",
        "height": 350,
        "series": [ ... ],
        "options": { ... }
      }
    },
    {
      "type": "chart",
      "gridSize": 6,
      "chartConfig": {
        "type": "pie",
        "title": "Team Contributions",
        "description": "Pie chart showing team contributions",
        "height": 350,
        "series": [ ... ],
        "options": { ... }
      }
    },
    {
      "type": "chart",
      "gridSize": 6,
      "chartConfig": {
        "type": "area",
        "title": "Stock Price Movements",
        "description": "Area chart showing stock price trends over time",
        "height": 350,
        "series": [ ... ],
        "options": { ... }
      }
    },
    {
      "type": "chart",
      "gridSize": 6,
      "chartConfig": {
        "type": "scatter",
        "title": "Sample Data Scatter",
        "description": "Scatter chart showing sample data points",
        "height": 350,
        "series": [ ... ],
        "options": { ... }
      }
    },
    {
      "type": "chart",
      "gridSize": 6,
      "chartConfig": {
        "type": "radar",
        "title": "Performance Metrics by Category",
        "description": "Radar chart comparing performance metrics across categories",
        "height": 350,
        "series": [ ... ],
        "options": { ... }
      }
    },
    {
      "type": "chart",
      "gridSize": 6,
      "chartConfig": {
        "type": "heatmap",
        "title": "Activity Heatmap by Day and Hour",
        "description": "Heatmap showing activity intensity across days and hours",
        "height": 350,
        "series": [ ... ],
        "options": { ... }
      }
    },
    {
      "type": "table",
      "gridSize": 6,
      "tableConfig": {
        "title": "Data Summary Table",
        "description": "Table summarizing key data points",
        "dense": true,
        "columns": [ ... ],
        "data": [ ... ]
      }
    },
    {
      "type": "stat",
      "gridSize": 3,
      "statConfig": {
        "title": "Key Metric",
        "value": "Value",
        "change": 12.5,
        "period": "vs last period"
      }
    }
  ]
}

Ensure your output JSON directly reflects the input data and user prompt—do not include any dummy values or generic labels. Replace all placeholder text with accurate, data-driven labels, values, and chart configurations based on the provided raw data and prompt.

Respond with ONLY the complete valid JSON configuration.
`;

		// const llmPrompt = `
		// 	You are a dashboard design expert with advanced data analysis skills. Based on the following prompt and raw data, create a JSON configuration for a dynamic dashboard with multiple components. Your goal is to provide meaningful insights by analyzing the data and determining the best way to process it (e.g., aggregating, grouping, or leaving it as is) based on the user's intent and the data's structure.

		// 	User Prompt: "${prompt}"

		// 	Raw Data: ${JSON.stringify({ sampleData }, null, 2)}

		// 	Your task:
		// 	1. Analyze the raw data and the user's prompt to determine the most insightful way to process it. This could include:
		// 		- Identifying logical groupings (e.g., by a recurring field like "Line," "SKU," or another key) if aggregation enhances insights.
		// 		- Calculating sums, averages, or other statistics for numeric fields based on their context (e.g., totals for counts, averages for rates).
		// 		- Preserving individual records if granularity is more valuable than aggregation.
		// 		- Highlighting trends, outliers, or comparisons that stand out.
		// 	2. Use your expertise to infer the meaning of fields (e.g., production totals, efficiency metrics, downtime, etc.) and prioritize those that drive decision-making.
		// 	3. Design a dashboard with a mix of components (charts, tables, stats) that:
		// 		- Visualizes key metrics and trends clearly.
		// 		- Compares performance across relevant categories.
		// 		- Highlights critical insights (e.g., top performers, bottlenecks, inefficiencies).
		// 		- Adapts dynamically to the data's structure and the user's intent

		// 	Respond with ONLY a valid JSON configuration for a dashboard with multiple components. The configuration should follow this format:

		// 	{
		// 		"title": "Dashboard Title",
		// 		"description": "Dashboard description",
		// 		"layout": "grid",
		// 		"components": [
		// 			{
		// 				"type": "chart",
		// 				"gridSize": 6,
		// 				"chartConfig": {
		// 					"type": "bar|line|pie|area|radar|scatter|heatmap",
		// 					"title": "Chart Title",
		// 					"description": "Brief description",
		// 					"height": 350,
		// 					"data": {
		// 						"categories": ["Category1", "Category2", ...],
		// 						"series": [
		// 							{
		// 								"name": "Series Name",
		// 								"data": [value1, value2, ...]
		// 							}
		// 						]
		// 					},
		// 					"xAxisTitle": "X-Axis Title",
		// 					"yAxisTitle": "Y-Axis Title",
		// 					"options": {}
		// 				}
		// 			},
		// 			{
		// 				"type": "table",
		// 				"gridSize": 6,
		// 				"tableConfig": {
		// 					"title": "Table Title",
		// 					"description": "Brief description",
		// 					"dense": true,
		// 					"columns": [
		// 						{
		// 							"header": "Column Name",
		// 							"field": "fieldName",
		// 							"align": "left|right|center"
		// 						}
		// 					],
		// 					"data": [
		// 						{
		// 							"fieldName": "value",
		// 							...
		// 						}
		// 					]
		// 				}
		// 			},
		// 			{
		// 				"type": "stat",
		// 				"gridSize": 3,
		// 				"statConfig": {
		// 					"title": "Stat Title",
		// 					"value": "Value",
		// 					"change": 12.5,
		// 					"period": "vs last period"
		// 				}
		// 			}
		// 		]
		// 	}

		// 	Ensure the dashboard provides meaningful insights tailored to the user's prompt and the data, with no predefined assumptions about aggregation.
		// `;

		// const llmPrompt = `
		// 	You are an expert in dashboard design and data analysis, tasked with creating a highly accurate and actionable dashboard configuration for a critical production environment. Based on the user's prompt and the provided raw data, generate a JSON configuration that delivers precise, error-free insights tailored to the user's intent.

		// 	User Prompt: "${prompt}"

		// 	Raw Data: ${JSON.stringify({ sampleData }, null, 2)}

		// 	Your task:
		// 	1. Thoroughly analyze the raw data and the user's prompt to determine the most effective way to process it for maximum insight and accuracy. Options include:
		// 		- Grouping or aggregating data by a field (e.g., "Line," "SKU Information," or another key) if it reveals trends or comparisons.
		// 		- Calculating sums, averages, or other statistics for numeric fields based on their semantic meaning (e.g., totals for production counts, averages for efficiency percentages).
		// 		- Preserving individual records if detailed granularity is more valuable.
		// 		- Identifying and highlighting trends, outliers, or performance gaps.
		// 	2. Special handling for duplicate fields:
		// 		- If a field like "Line" has multiple entries with different values in another field (e.g., "SKU Information"), treat each entry as distinct rather than aggregating or skipping them.
		// 		- When confusion arises (e.g., multiple "Line" values in a "Line vs OEE" chart), list all relevant entries separately and include the differentiating factor (e.g., "SKU Information") in labels or descriptions to clarify for the user.
		// 	3. Infer the meaning of each field (e.g., production output, efficiency, downtime, reliability) and prioritize those critical for decision-making in a production context.
		// 	4. Design a dashboard with a mix of components (charts, tables, stats) that:
		// 		- Visualizes key metrics with precision and clarity.
		// 		- Enables comparisons across relevant categories (e.g., lines, SKUs, or time periods), ensuring all distinct entries are represented when needed.
		// 		- Highlights actionable insights (e.g., inefficiencies, bottlenecks, top performers).
		// 		- Adapts dynamically to the data structure and prompt without assumptions.
		// 	5. Ensure the output is error-free, avoids generic placeholders, and directly reflects the provided data.

		// 	Respond with ONLY a valid JSON configuration for a dashboard with multiple components, following this format:

		// 	{
		// 		"title": "Dashboard Title",
		// 		"description": "Dashboard description",
		// 		"layout": "grid",
		// 		"components": [
		// 			{
		// 				"type": "chart",
		// 				"gridSize": 6,
		// 				"chartConfig": {
		// 					"type": "bar|line|pie|area|radar|scatter|heatmap",
		// 					"title": "Chart Title",
		// 					"description": "Brief description",
		// 					"height": 350,
		// 					"data": {
		// 						"categories": ["Category1", "Category2", ...],
		// 						"series": [
		// 							{
		// 								"name": "Series Name",
		// 								"data": [value1, value2, ...]
		// 							}
		// 						]
		// 					},
		// 					"xAxisTitle": "X-Axis Title",
		// 					"yAxisTitle": "Y-Axis Title",
		// 					"options": {}
		// 				}
		// 			},
		// 			{
		// 				"type": "table",
		// 				"gridSize": 6,
		// 				"tableConfig": {
		// 					"title": "Table Title",
		// 					"description": "Brief description",
		// 					"dense": true,
		// 					"columns": [
		// 						{
		// 							"header": "Column Name",
		// 							"field": "fieldName",
		// 							"align": "left|right|center"
		// 						}
		// 					],
		// 					"data": [
		// 						{
		// 							"fieldName": "value",
		// 							...
		// 						}
		// 					]
		// 				}
		// 			},
		// 			{
		// 				"type": "stat",
		// 				"gridSize": 3,
		// 				"statConfig": {
		// 					"title": "Stat Title",
		// 					"value": "Value",
		// 					"change": 12.5,
		// 					"period": "vs last period"
		// 				}
		// 			}
		// 		]
		// 	}

		// 	The dashboard must be precise, actionable, and directly tied to the input data and prompt—no dummy data or placeholders allowed.
		// `;
		// const llmPrompt = `
		// 	You are an expert in dashboard design and data analysis, tasked with creating a highly accurate, actionable, and visually diverse dashboard configuration for a critical production environment. Based on the user's prompt and the provided raw data, generate a JSON configuration that delivers precise, error-free insights tailored to the user's intent, utilizing a variety of chart types and the full breadth of the data.

		// 	User Prompt: "${prompt}"

		// 	Raw Data: ${JSON.stringify({ sampleData }, null, 2)}

		// 	Your task:
		// 	1. Thoroughly analyze the raw data and the user's prompt to determine the most effective way to process it for maximum insight and accuracy. Options include:
		// 		- Grouping or aggregating data by a field (e.g., "Line," "SKU Information," or another key) if it reveals trends or comparisons.
		// 		- Calculating sums, averages, or other statistics for numeric fields based on their semantic meaning (e.g., totals for production counts, averages for efficiency percentages).
		// 		- Preserving individual records if detailed granularity is more valuable.
		// 		- Identifying and highlighting trends, outliers, or performance gaps.
		// 	2. Special handling for duplicate fields:
		// 		- If a field like "Line" has multiple entries with different values in another field (e.g., "SKU Information"), treat each entry as distinct rather than aggregating or skipping them.
		// 		- When confusion arises (e.g., in a "Line vs OEE" chart), list all relevant entries separately and include the differentiating factor (e.g., "SKU Information") in labels or descriptions for clarity.
		// 	3. Explore the full dataset:
		// 		- Analyze all available fields (e.g., production output, efficiency metrics, downtime categories, stoppages, reliability) and prioritize those critical for decision-making in a production context.
		// 		- Don’t limit focus to a few fields—use the richness of the data to provide a comprehensive view.
		// 	4. Design a dashboard with a diverse mix of components (charts, tables, stats) that:
		// 		- Uses a variety of chart types (bar, pie, scatter, line, area, radar, heatmap) appropriately matched to the data:
		// 			- Bar charts for comparisons (e.g., OEE or Final FG across lines/SKUs).
		// 			- Pie charts for proportions (e.g., breakdown of downtime categories).
		// 			- Scatter charts for correlations (e.g., OEE vs. downtime).
		// 			- Line charts for trends (if time data were present).
		// 		- Visualizes key metrics with precision and clarity across multiple dimensions (e.g., production, efficiency, losses).
		// 		- Enables comparisons across relevant categories (e.g., lines, SKUs), ensuring all distinct entries are represented.
		// 		- Highlights actionable insights (e.g., inefficiencies, bottlenecks, top performers).
		// 		- Adapts dynamically to the data structure and prompt without assumptions.
		// 	5. Ensure the output is error-free, avoids generic placeholders, and directly reflects the provided data.

		// 	Respond with ONLY a valid JSON configuration for a dashboard with multiple components, following this format:

		// 	{
		// 		"title": "Dashboard Title",
		// 		"description": "Dashboard description",
		// 		"layout": "grid",
		// 		"components": [
		// 			{
		// 				"type": "chart",
		// 				"gridSize": 6,
		// 				"chartConfig": {
		// 					"type": "bar|line|pie|area|radar|scatter|heatmap",
		// 					"title": "Chart Title",
		// 					"description": "Brief description",
		// 					"height": 350,
		// 					"data": {
		// 						"categories": ["Category1", "Category2", ...],
		// 						"series": [
		// 							{
		// 								"name": "Series Name",
		// 								"data": [value1, value2, ...]
		// 							}
		// 						]
		// 					},
		// 					"xAxisTitle": "X-Axis Title",
		// 					"yAxisTitle": "Y-Axis Title",
		// 					"options": {}
		// 				}
		// 			},
		// 			{
		// 				"type": "table",
		// 				"gridSize": 6,
		// 				"tableConfig": {
		// 					"title": "Table Title",
		// 					"description": "Brief description",
		// 					"dense": true,
		// 					"columns": [
		// 						{
		// 							"header": "Column Name",
		// 							"field": "fieldName",
		// 							"align": "left|right|center"
		// 						}
		// 					],
		// 					"data": [
		// 						{
		// 							"fieldName": "value",
		// 							...
		// 						}
		// 					]
		// 				}
		// 			},
		// 			{
		// 				"type": "stat",
		// 				"gridSize": 3,
		// 				"statConfig": {
		// 					"title": "Stat Title",
		// 					"value": "Value",
		// 					"change": 12.5,
		// 					"period": "vs last period"
		// 				}
		// 			}
		// 		]
		// 	}

		// 	The dashboard must be precise, actionable, visually diverse, and directly tied to the input data and prompt—no dummy data or placeholders allowed.
		// `;
		// const llmPrompt = `
		// 	You are an expert in dashboard design and data analysis, tasked with creating a highly accurate, actionable, and visually diverse dashboard configuration for a critical production environment. Based on the user's prompt and the provided raw data, generate a JSON configuration that delivers precise, error-free insights tailored to the user's intent, utilizing a variety of chart types and the full breadth of the data.

		// 	User Prompt: "${prompt}"

		// 	Raw Data: ${JSON.stringify({ sampleData }, null, 2)}

		// 	Your task:
		// 	1. Thoroughly analyze the raw data and the user's prompt to determine the most effective way to process it for maximum insight and accuracy. Options include:
		// 		- Grouping or aggregating data by a field (e.g., "Line," "SKU Information," or another key) if it reveals trends or comparisons.
		// 		- Calculating sums, averages, or other statistics for numeric fields based on their semantic meaning (e.g., totals for production counts, averages for efficiency percentages).
		// 		- Preserving individual records if detailed granularity is more valuable.
		// 		- Identifying and highlighting trends, outliers, or performance gaps.
		// 	2. Special handling for duplicate fields:
		// 		- If a field like "Line" has multiple entries with different values in another field (e.g., "SKU Information"), treat each entry as distinct rather than aggregating or skipping them.
		// 		- When confusion arises (e.g., in a "Line vs OEE" chart), list all relevant entries separately and include the differentiating factor (e.g., "SKU Information") in labels or descriptions for clarity.
		// 	3. Explore the full dataset:
		// 		- Analyze all available fields (e.g., production output, efficiency metrics, downtime categories, stoppages, reliability) and prioritize those critical for decision-making in a production context.
		// 		- Don’t limit focus to a few fields—use the richness of the data to provide a comprehensive view.
		// 	4. Design a dashboard with a diverse mix of components (charts, tables, stats) that:
		// 		- Uses a variety of chart types (bar, pie, scatter, line, area, radar, heatmap) appropriately matched to the data:
		// 			- Bar charts for comparisons (e.g., OEE or Final FG across lines/SKUs).
		// 			- Pie charts for proportions (e.g., breakdown of downtime categories), ensuring categories are meaningful, non-zero, and correctly labeled (e.g., "Unbooked Minor Downtime," "Quality Loss / Checks").
		// 			- Scatter charts for correlations (e.g., OEE vs. downtime).
		// 			- Line charts for trends (if time data were present).
		// 		- Visualizes key metrics with precision and clarity across multiple dimensions (e.g., production, efficiency, losses).
		// 		- Enables comparisons across relevant categories (e.g., lines, SKUs), ensuring all distinct entries are represented.
		// 		- Highlights actionable insights (e.g., inefficiencies, bottlenecks, top performers).
		// 		- Adapts dynamically to the data structure and prompt without assumptions.
		// 	5. Ensure chart data integrity:
		// 		- For pie charts (or any chart), only include non-zero values and ensure categories are explicitly labeled with the corresponding field names (e.g., "Unbooked Minor Downtime" instead of "series-1").
		// 		- If a chart cannot be meaningfully generated (e.g., no non-zero data for a pie chart), replace it with a more suitable component like a stat or table summarizing the data.
		// 	6. Ensure the output is error-free, avoids generic placeholders (e.g., "series-1"), and directly reflects the provided data.

		// 	Respond with ONLY a valid JSON configuration for a dashboard with multiple components, following this format:

		// 	{
		// 		"title": "Dashboard Title",
		// 		"description": "Dashboard description",
		// 		"layout": "grid",
		// 		"components": [
		// 			{
		// 				"type": "chart",
		// 				"gridSize": 6,
		// 				"chartConfig": {
		// 					"type": "bar|line|pie|area|radar|scatter|heatmap",
		// 					"title": "Chart Title",
		// 					"description": "Brief description",
		// 					"height": 350,
		// 					"data": {
		// 						"categories": ["Category1", "Category2", ...],
		// 						"series": [
		// 							{
		// 								"name": "Series Name",
		// 								"data": [value1, value2, ...]
		// 							}
		// 						]
		// 					},
		// 					"xAxisTitle": "X-Axis Title",
		// 					"yAxisTitle": "Y-Axis Title",
		// 					"options": {}
		// 				}
		// 			},
		// 			{
		// 				"type": "table",
		// 				"gridSize": 6,
		// 				"tableConfig": {
		// 					"title": "Table Title",
		// 					"description": "Brief description",
		// 					"dense": true,
		// 					"columns": [
		// 						{
		// 							"header": "Column Name",
		// 							"field": "fieldName",
		// 							"align": "left|right|center"
		// 						}
		// 					],
		// 					"data": [
		// 						{
		// 							"fieldName": "value",
		// 							...
		// 						}
		// 					]
		// 				}
		// 			},
		// 			{
		// 				"type": "stat",
		// 				"gridSize": 3,
		// 				"statConfig": {
		// 					"title": "Stat Title",
		// 					"value": "Value",
		// 					"change": 12.5,
		// 					"period": "vs last period"
		// 				}
		// 			}
		// 		]
		// 	}

		// 	The dashboard must be precise, actionable, visually diverse, and directly tied to the input data and prompt—no dummy data, placeholders, or generic labels allowed.
		// `;

		// const { data } = sampleData;
		let formattedData = '';
		if (Array.isArray(data) && data.length > 0) {
			const headers = Object.keys(data[0]);
			formattedData = headers.join(', ') + '\n';
			formattedData += data.map((row) => headers.map((header) => row[header] ?? '').join(', ')).join('\n');
		} else {
			formattedData = 'No data available.';
		}

		// const llmPrompt = `zz
		// 	You are an expert in dashboard design and data analysis, tasked with creating a highly accurate, actionable, and visually diverse dashboard configuration for a critical production environment. Based on the user's prompt and the provided raw data, generate a JSON configuration for a dashboard with multiple components (charts, tables, stats) that delivers precise, error-free insights tailored to the user's intent.

		// 	User Prompt: "${prompt}"

		// 	Raw Data (in CSV-like format):
		// 	${formattedData}

		// 	Your task:
		// 	1. Analyze the raw data and the user's prompt to determine the most effective way to process it for maximum insight and accuracy. Options include:
		// 		- Grouping or aggregating data by a field (e.g., a categorical column like "department" or "Line") if it reveals trends or comparisons.
		// 		- Calculating sums, averages, or other statistics for numeric fields based on their semantic meaning (e.g., totals for counts, averages for percentages).
		// 		- Preserving individual records if detailed granularity is more valuable.
		// 		- Identifying and highlighting trends, outliers, or performance gaps.
		// 	2. Special handling for duplicate fields:
		// 		- If a field (e.g., "Line") has multiple entries with different values in another field (e.g., "SKU Information"), treat each entry as distinct rather than aggregating or skipping them.
		// 		- When confusion arises (e.g., in a "Line vs OEE" chart), list all relevant entries separately and include the differentiating factor (e.g., "SKU Information") in labels or descriptions for clarity.
		// 	3. Explore the full dataset:
		// 		- Analyze all available fields and prioritize those critical for decision-making (e.g., production output, efficiency metrics, downtime, etc.).
		// 		- Don’t limit focus to a few fields—use the richness of the data to provide a comprehensive view.
		// 	4. Design a dashboard with a diverse mix of components (charts, tables, stats) that:
		// 		- Includes at least 3-5 components to cover different aspects of the data (e.g., comparisons, proportions, correlations, summaries).
		// 		- Uses a variety of chart types (bar, pie, scatter, line, area, radar, heatmap) appropriately matched to the data:
		// 			- Bar charts for comparisons (e.g., numerical values across categories like "sales by department").
		// 			- Pie charts for proportions (e.g., breakdown of downtime categories), ensuring categories are meaningful, non-zero, and correctly labeled.
		// 			- Scatter charts for correlations (e.g., two numerical fields like "OEE vs downtime").
		// 			- Line charts for trends (if a time-based or sequential column is present).
		// 		- Visualizes key metrics with precision and clarity across multiple dimensions.
		// 		- Enables comparisons across relevant categories, ensuring all distinct entries are represented.
		// 		- Highlights actionable insights (e.g., inefficiencies, bottlenecks, top performers).
		// 		- Adapts dynamically to the data structure and prompt without assumptions.

		// 	5. Ensure chart data integrity:
		// 		- For all charts, only include non-zero values and ensure categories are explicitly labeled with the corresponding field names (e.g., "Unbooked Minor Downtime" instead of "series-1").
		// 		- For pie charts, map categorical labels to field names and numerical values to non-zero data points.
		// 		- If a chart cannot be meaningfully generated (e.g., no non-zero data for a pie chart, or no time data for a line chart), replace it with a more suitable component like a stat or table summarizing the data.
		// 	6. Handle dynamic functions in JSON:
		// 		- Do NOT include JavaScript functions directly in the JSON output (e.g., "formatter": function() {...}).
		// 		- Instead, if a dynamic function is needed (e.g., for a chart's formatter), include it as a string representation of the function (e.g., "formatter": "function() { return this.point.name + ': ' + this.percentage.toFixed(1) + '%'; }").
		// 		- Ensure all other values in the JSON are valid JSON types: strings, numbers, objects, arrays, booleans, or null.
		// 	7. Ensure the output is error-free, avoids generic placeholders (e.g., "series-1"), and directly reflects the provided data.
		// 	8. For chart use accurate apex chart config.
		// 	{
		// 		"title": "Dashboard Title",
		// 		"description": "Dashboard description",
		// 		"layout": "grid",
		// 		"components": [
		// 			{
		// 				"type": "chart",
		// 				"gridSize": 6,
		// 				"chartConfig": {
		// 					"use accurate apex chart config here "
		// 			},
		// 			{
		// 				"type": "table",
		// 				"gridSize": 6,
		// 				"tableConfig": {
		// 					"title": "Table Title",
		// 					"description": "Brief description",
		// 					"dense": true,
		// 					"columns": [
		// 						{
		// 							"header": "Column Name",
		// 							"field": "fieldName",
		// 							"align": "left|right|center"
		// 						}
		// 					],
		// 					"data": [
		// 						{
		// 							"fieldName": "value",
		// 							...
		// 						}
		// 					]
		// 				}
		// 			},
		// 			{
		// 				"type": "stat",
		// 				"gridSize": 3,
		// 				"statConfig": {
		// 					"title": "Stat Title",
		// 					"value": "Value",
		// 					"change": 12.5,
		// 					"period": "vs last period"
		// 				}
		// 			}
		// 		]
		// 	}

		// 	The dashboard must be precise, actionable, visually diverse, and directly tied to the input data and prompt—no dummy data, placeholders, generic labels, or direct JavaScript functions allowed.
		// `;

		const initialMessages = [{ role: 'user', content: llmPrompt2 }];
		const userMessage = { role: 'user', content: `${prompt}` };

		// Use generateText (non-streaming) for Config Mode
		const response = await generateText({
			// model: google('gemini-2.5-pro-exp-03-25'),
			model: google('gemini-2.0-flash-001'),
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

		// console.log('dashboardConfig', dashboardConfig);
		// dashboardConfig.components.forEach((component) => {
		// 	if (component.type === 'chart' && component.chartConfig && component.chartConfig.options) {
		// 		const options = component.chartConfig.options;
		// 		if (options.dataLabels && options.dataLabels.formatter && typeof options.dataLabels.formatter === 'string') {
		// 			try {
		// 				// Convert the stringified function into an actual function
		// 				options.dataLabels.formatter = new Function('return ' + options.dataLabels.formatter)();
		// 			} catch (evalError) {
		// 				console.error('Failed to evaluate formatter function:', evalError);
		// 				// Fallback to a default formatter if evaluation fails
		// 				options.dataLabels.formatter = (val, opts) => `${opts.w.globals.labels[opts.seriesIndex]}: ${val.toFixed(1)}%`;
		// 			}
		// 		}
		// 	}
		// });
		// const cleanedText = resultText
		// 	.replace(/```json/g, '')
		// 	.replace(/```/g, '')
		// 	.trim();

		return c.json({ success: true, data: data, dashboardConfig, prompt });
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
