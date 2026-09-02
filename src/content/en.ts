import type { LocalePack } from './index'

/**
 * English text and UI strings.
 *
 * Rules (see docs/plan.md)
 * - executive: no equations or symbols; analogies and outcomes. At most 3 paragraphs.
 * - engineer: precise terms and equations, but only facts with a verified source (the original paper etc.).
 * - Concrete numbers only when the source is confirmed.
 * - Keep paragraph and key-term counts equal to ko.ts (content.test.ts checks this).
 */
export const en: LocalePack = {
  ui: {
    documentTitle: 'Transformers, explained simply',
    brand: 'Transformers, explained simply',
    skipToContent: 'Skip to content',
    audienceLegend: 'Level of detail',
    audience: {
      executive: { label: 'Executive', hint: 'analogies, no math' },
      engineer: { label: 'Engineer', hint: 'equations and structure' },
    },
    localeLegend: 'Language',
    tocTitle: 'Contents',
    hero: {
      kicker: 'The architecture behind AI like ChatGPT',
      title: 'Transformers, explained simply',
      lead: 'We start with analogies and no math, and go down to the equations and structure only if you want to. Change the level of detail and the language at the top right.',
      modeBefore: 'You are reading the ',
      modeAfter: {
        executive: ' version. Each section shows only the analogy and the key points.',
        engineer: ' version. Each section ends with equations and structural detail.',
      },
    },
    analogyTitle: 'Think of it this way',
    engineerTitle: 'Going deeper for engineers',
    draftBadge: 'Draft',
    tokenizer: {
      heading: 'Try it: split a sentence into tokens',
      description: 'Edit the sentence and watch how it splits into pieces (tokens).',
      inputLabel: 'Sentence to tokenize',
      listLabel: 'Token list',
      count: (n) => `${n} token${n === 1 ? '' : 's'}`,
      note: 'To keep the idea visible, this demo splits only on spaces and punctuation. Real models use subword tokenizers, so the number of pieces and their boundaries can differ.',
      defaultText: 'I went to the bank to withdraw money.',
    },
    flow: {
      heading: 'At a glance: the six steps that produce the next word',
      description:
        'Click a step, or use the "Next step" button, to follow the whole flow. Each step corresponds to one section below.',
      prev: 'Previous step',
      next: 'Next step',
      stepLabel: (index, total) => `Step ${index} of ${total}`,
      goToSection: 'Read more about this step',
      stages: [
        {
          section: 'tokens',
          label: 'Split into tokens',
          detail:
            'Cut the sentence into pieces (tokens) and turn each piece into a list of numbers (a vector).',
        },
        {
          section: 'position',
          label: 'Add order',
          detail: 'Add numbers to each token that encode which position it is in.',
        },
        {
          section: 'attention',
          label: 'Consult each other',
          detail:
            'Attention. Each token looks around at the other tokens and pulls in the information that matters to its meaning.',
        },
        {
          section: 'ffn',
          label: 'Think it over alone',
          detail:
            'The feed-forward layer. Each token digests what it collected without looking at any other token.',
        },
        {
          section: 'stack',
          label: 'Repeat in layers',
          detail:
            'Steps 3 and 4 form a layer. Stacking many layers folds in progressively deeper context.',
        },
        {
          section: 'output',
          label: 'Draw the next word',
          detail:
            'Compute a probability for every word in the vocabulary, draw one, append it to the text, and start again from step 1.',
        },
      ],
    },
    temperature: {
      heading: 'Try it: change the temperature and draw the next word',
      description:
        'See how temperature acts when the model turns its scores into probabilities. Move the slider, then press a draw button.',
      context: 'Every morning I drink a cup of ___',
      sliderLabel: 'Temperature',
      valueLabel: (temperature) => `Temperature ${temperature.toFixed(1)}`,
      draw: 'Draw once',
      drawMany: 'Draw 10 times',
      resultHeading: 'Draws',
      tableCaption: 'Probability of each candidate word',
      columns: { rank: 'Rank', token: 'Candidate', probability: 'Probability' },
      candidates: [
        { token: 'coffee', score: 3.0 },
        { token: 'tea', score: 2.2 },
        { token: 'water', score: 1.6 },
        { token: 'juice', score: 1.2 },
        { token: 'milk', score: 0.8 },
        { token: 'cocoa', score: 0.2 },
        { token: 'soup', score: -0.3 },
        { token: 'wine', score: -1.0 },
      ],
      regimes: {
        low: 'At low temperature the probability piles onto the top word. Answers are stable, but the model keeps saying the same thing.',
        neutral: 'Temperature 1.0 uses the probabilities exactly as the model computed them.',
        high: 'At high temperature the probability spreads out evenly. More variety, but odd words can be drawn.',
      },
      note: 'The candidate words and scores are made up for this explanation and are not the output of a real model. The computation that turns scores into probabilities (softmax with temperature) is the real one.',
    },
    positional: {
      heading: 'Try it: a different pattern for every position',
      description:
        "The original paper's sine and cosine positional encoding. Each row is one position, and no two rows share a pattern. Click a row name to see the numbers for that position.",
      rowLabel: (position) => `Position ${position}`,
      columnLabel: (dimension) => `Dimension ${dimension}`,
      selectedHeading: (position) => `Pattern for position ${position}`,
      legendLabel: 'Value: blue is -1, gray is 0, red is +1',
      tableCaption: 'Encoding values by position (rows: positions, columns: dimensions)',
      note: 'The dimension is reduced to 16 for this explanation (the original paper uses 512). The formula is the one in section 3.5 of the paper. Dimensions on the left change quickly and those on the right change slowly.',
    },
    attention: {
      heading: 'Try it: follow the attention computation step by step',
      description:
        'A small example with four tokens. Pick which token to look through, and see how much it attends to each of the others. The computation can be expanded below.',
      sentence: 'I went to the bank to withdraw money',
      tokens: ['went', 'bank', 'withdraw', 'money'],
      q: [
        [3, 1, 0],
        [1, -1, 4],
        [1, 0, 3],
        [4, -1, 1],
      ],
      k: [
        [0, 1, 0],
        [1, 0, 0],
        [0.5, 0.5, 0.5],
        [0, 0, 1],
      ],
      v: [
        [0, 1],
        [0.5, 0.5],
        [0.5, 0.5],
        [1, 0],
      ],
      valueDims: ['finance component', 'action component'],
      queryLegend: 'Which token are we looking through?',
      causalLabel: 'Causal mask: hide the tokens that come later (as in a decoder-only model)',
      weightsHeading: (token) => `Weights as seen from "${token}"`,
      masked: 'hidden',
      maskedNote: (token) =>
        `Tokens after "${token}" are hidden. In a decoder-only model, the later tokens are the ones that look back at earlier ones.`,
      outputHeading: (token) => `New vector for "${token}"`,
      detailsSummary: 'Show the computation',
      steps: {
        vectors: '1. Q (query), K (key), and V (value) vectors of each token',
        scores: '2. Score = Q · K (dot product)',
        scaled: '3. Divide by √d_k (d_k = 3, so √3 ≈ 1.73)',
        softmax: '4. softmax turns the scores into weights that sum to 1',
        output: '5. Sum weight × V over all tokens to get the new vector',
      },
      columns: {
        token: 'Token',
        q: 'Q',
        k: 'K',
        v: 'V',
        score: 'Score',
        scaled: 'Scaled',
        weight: 'Weight',
      },
      note: 'The numbers in Q, K, and V were chosen to show the computation; they are not values learned by a real model. The names of the two components are for explanation only, since real model dimensions carry no such names. The computation is the scaled dot-product attention of the original paper.',
    },
    ffn: {
      heading: 'Try it: each token separately, but the same computation',
      description:
        'After attention, every token vector passes through the same two-layer network. Pick a token to see its computation. Tokens do not look at each other in this step.',
      tokens: ['went', 'bank', 'withdraw', 'money'],
      inputs: [
        [0.8, 0.2],
        [0.1, 0.9],
        [1.0, 0.1],
        [0.5, 0.6],
      ],
      w1: [
        [1.0, -1.0, 0.5, 0.0],
        [0.0, 1.0, -0.5, 1.0],
      ],
      b1: [0, -0.5, 0, -0.2],
      w2: [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [-0.5, 0.5],
      ],
      b2: [0, 0],
      selectLegend: 'Which token do you want to follow?',
      sharedLabel:
        'All four tokens use the same weights (W₁, b₁, W₂, b₂). The key point is that there are no connections between tokens.',
      stageLabels: {
        input: 'Input x',
        preActivation: 'xW₁ + b₁',
        hidden: 'After ReLU (negatives become 0)',
        output: 'Output = hidden · W₂ + b₂',
      },
      zeroNote: 'Hidden values that became 0 are negatives cut off by ReLU.',
      note: 'The numbers are illustrative. Real models have far larger dimensions (original paper: 512 → 2048 → 512) and their values are set by training.',
    },
    stack: {
      heading: 'Try it: stack the layers',
      description:
        'One layer consists of attention and a feed-forward step; each result is added back onto its input and normalized (a residual connection). Change the number of layers.',
      sliderLabel: 'Layers',
      layerLabel: (count) => `${count} ${count === 1 ? 'layer' : 'layers'}`,
      layersCaption: (count) =>
        `${count} ${count === 1 ? 'layer' : 'layers'}. The original paper used 6 layers each in the encoder and the decoder, and larger models use many more.`,
      layerTitle: (index) => `Layer ${index}`,
      attention: 'Attention',
      ffn: 'Feed-forward',
      norm: 'Norm',
      residual:
        'The line that arcs over each box is the residual connection: the input skips the computation and is added back in.',
      inputLabel: 'Token vectors (embedding + position)',
      outputLabel: 'To the next-word probabilities',
      note: 'Each layer has its own parameters. The drawing follows the original paper (add, then normalize); GPT-2 and others place the normalization at the input of each sub-layer.',
    },
    training: {
      heading: 'Try it: predict, score, adjust',
      description:
        'Play "guess the next word" on a sentence whose answer is already written. Each press of a training button corrects the scores by the size of the mistake. Watch the probability of the answer rise and the loss fall.',
      context: 'Every morning I drink a cup of ___',
      answerLabel: 'answer',
      answer: 'coffee',
      candidates: [
        { token: 'tea', score: 2.0 },
        { token: 'water', score: 1.5 },
        { token: 'coffee', score: 1.0 },
        { token: 'juice', score: 0.5 },
        { token: 'milk', score: 0.0 },
      ],
      stepOnce: 'Train once',
      stepMany: 'Train 10 times',
      reset: 'Reset',
      stepsLabel: 'Training steps',
      lossLabel: 'Loss (size of the mistake)',
      answerProbabilityLabel: 'Probability of the answer',
      columns: { token: 'Candidate', probability: 'Probability' },
      correctMark: 'answer',
      tableCaption: 'Current probability of each candidate',
      note: 'The loss is -log of the probability of the answer (cross-entropy), and each adjustment moves the scores a little along its gradient. A real model adjusts its internal parameters rather than the scores directly, but the direction is the same. Candidates and scores are illustrative.',
    },
    summary: {
      cardTitle: 'One-minute summary',
      steps: [
        'Split the sentence into tokens and turn each into a list of numbers',
        'Add position information',
        'Let tokens consult each other with attention, then process on their own with the feed-forward layer',
        'Repeat this layer many times',
        "Compute the next token's probabilities, draw one, and repeat",
      ],
      takeawaysTitle: 'Three points that help with decisions',
      takeaways: [
        {
          lead: 'Long context is expensive.',
          text: 'Attention cost grows with the square of the input length.',
        },
        {
          lead: 'Performance depends on three things.',
          text: 'The amount and quality of data, model size, and compute.',
        },
        {
          lead: 'Settings alone change the output.',
          text: 'Generation settings such as temperature change the character of the answers.',
        },
      ],
      nextStepsTitle: 'Next steps for engineers',
      nextSteps: [
        {
          label: 'Read the original paper',
          detail:
            'Vaswani et al., "Attention Is All You Need" (2017). Section 3.2 (attention), 3.3 (feed-forward), 3.5 (positional encoding).',
          url: 'https://arxiv.org/abs/1706.03762',
        },
        {
          label: "Read this page's small implementation",
          detail:
            'softmax, scaled dot-product attention, causal mask, and positional encoding in src/lib/math.ts, with tests.',
          url: 'https://github.com/yhkaichase/transformer_explain',
        },
        {
          label: 'Look inside a real GPT-2',
          detail: 'Transformer Explainer runs GPT-2 in the browser and shows its attention values.',
          url: 'https://poloclub.github.io/transformer-explainer/',
        },
        {
          label: 'Read the GPT-2 model code',
          detail:
            'Learned position embeddings, normalization at sub-layer inputs, and tied embeddings, all visible in the code.',
          url: 'https://github.com/openai/gpt-2/blob/master/src/model.py',
        },
      ],
    },
    sourcesTitle: 'References',
  },

  sources: [
    {
      label: 'Vaswani et al., "Attention Is All You Need" (2017)',
      detail:
        'The paper that introduced the Transformer. The basis for the equations and structure described on this page.',
      url: 'https://arxiv.org/abs/1706.03762',
    },
    {
      label: 'Transformer Explainer (Georgia Tech Polo Club)',
      detail:
        'An interactive visualization that runs a real GPT-2 in the browser and shows its internals. The starting point for this project.',
      url: 'https://github.com/poloclub/transformer-explainer',
    },
    {
      label: 'OpenAI, GPT-2 model code (src/model.py)',
      detail:
        'The basis for the learned position embeddings (wpe), normalization at sub-layer inputs, weight sharing between the output layer and the embedding, and the GELU activation.',
      url: 'https://github.com/openai/gpt-2/blob/master/src/model.py',
    },
    {
      label:
        'Shaw, Uszkoreit, Vaswani, "Self-Attention with Relative Position Representations" (2018)',
      detail: 'Relative position information inside the attention computation.',
      url: 'https://arxiv.org/abs/1803.02155',
    },
    {
      label: 'He et al., "Deep Residual Learning for Image Recognition" (2016)',
      detail: 'The paper that proposed residual connections.',
      url: 'https://arxiv.org/abs/1512.03385',
    },
    {
      label: 'Fan, Lewis, Dauphin, "Hierarchical Neural Story Generation" (2018)',
      detail: 'Top-k sampling.',
      url: 'https://arxiv.org/abs/1805.04833',
    },
    {
      label: 'Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020)',
      detail: 'Top-p (nucleus) sampling.',
      url: 'https://arxiv.org/abs/1904.09751',
    },
    {
      label:
        'Ouyang et al., "Training language models to follow instructions with human feedback" (2022)',
      detail: 'InstructGPT: the supervised fine-tuning and RLHF pipeline.',
      url: 'https://arxiv.org/abs/2203.02155',
    },
  ],

  sections: {
    intro: {
      title: 'The Transformer in one sentence',
      shortTitle: 'One sentence',
      tagline:
        'A machine in which the words of a sentence consult one another to guess the next word.',
      analogy:
        'Picture the words of a sentence sitting in a meeting room. To pin down its own meaning, each word looks around at the others and listens more closely to the ones that matter to it. After repeating that meeting several times, the room votes on the next word.',
      executive: [
        'The Transformer is an AI architecture proposed by Google researchers in the 2017 paper "Attention Is All You Need". The GPT in ChatGPT stands for "Generative Pre-trained Transformer", so the architecture is right there in the name. Most of today\'s conversational AI, translation, and code-generation tools are built on it.',
        'What it does is surprisingly simple: it reads the text so far and predicts the next word. Repeat that one word at a time and you get a sentence, then a paragraph, then an answer.',
        'The key mechanism is called "attention". It lets every word work out for itself which other words in the sentence matter to it. This page walks through that process step by step.',
      ],
      engineer: [
        "The original paper's architecture was an encoder-decoder built for translation: an encoder reads the input sentence and a decoder produces the output. GPT-style models instead stack only decoder blocks, a decoder-only architecture. This page follows the decoder-only flow because it is simpler to explain.",
        'The overall flow is: token sequence → embedding + position information → [attention → feed-forward] × N layers → a probability distribution over the next token. The sections that follow trace exactly this order.',
      ],
      keyTerms: [
        {
          term: 'Token',
          meaning:
            'The smallest unit of text the model works with. It can be a whole word or a piece of one.',
        },
        {
          term: 'Parameter',
          meaning:
            'The numbers inside the model that training adjusts. A "7-billion-parameter model" has 7 billion of them.',
        },
      ],
      status: 'ready',
    },

    tokens: {
      title: 'Turning text into numbers: tokens and embeddings',
      shortTitle: 'Tokens and embeddings',
      tagline:
        'Computers cannot do arithmetic on letters, so a sentence is cut into pieces and each piece becomes a list of numbers.',
      analogy:
        'Imagine attaching a "personality card" to every word. Each card has hundreds of dials, and words with similar meanings end up with similar dial patterns. That card is the embedding.',
      executive: [
        'Step one is splitting the sentence into small pieces called tokens. Type a sentence below to see how it is split.',
        'Step two turns each token into a list of numbers, a vector. Nobody sets these numbers by hand. During training they settle into place so that words used in similar contexts get similar numbers.',
        'One thing matters here: there is no context yet. The token "bank" receives the same card whether it means a place that holds money or the side of a river. Context is added in the next step, attention.',
      ],
      engineer: [
        "Real models do not split on spaces; they use a subword tokenizer. Frequent words stay whole and rare words are broken into several pieces. Byte Pair Encoding (BPE) is the best-known method. The demo above splits only on spaces and punctuation to keep the idea visible, so its tokens differ from a real model's.",
        'The embedding is a trainable lookup table of size vocabulary × model dimension (d_model). A token ID simply selects one row, and those values are updated by backpropagation during training.',
      ],
      keyTerms: [
        {
          term: 'Vector',
          meaning:
            'A list of numbers in a row. Inside the model, one token is represented by one vector.',
        },
        {
          term: 'd_model',
          meaning:
            'The length of each vector. A larger value lets each token carry a richer representation, at the cost of more computation.',
        },
      ],
      status: 'ready',
    },

    position: {
      title: 'Remembering order: position information',
      shortTitle: 'Position',
      tagline:
        'Attention looks at all the words at once, so the model must be told separately which word came where.',
      analogy:
        'It is like writing a seat number on each participant\'s name tag. You need to know who sits where before you can refer to "what the person on my left said".',
      executive: [
        '"Dog bites man" and "Man bites dog" contain the same words but mean the opposite. Order changes meaning.',
        'A Transformer does not read a sentence one word at a time; it sees every word simultaneously. So it adds numbers that encode "which position am I in" to each token\'s vector. This is called positional encoding.',
      ],
      engineer: [
        'The original paper added fixed positional encodings built from sine and cosine functions to the token embeddings. Waves of different periods are layered so that every position gets a unique pattern.',
        'GPT-2 learns its position embeddings (the wpe variable in the published model code). The original paper also tried learned position embeddings and reported nearly identical results. There are also methods that inject the relative distance between tokens inside the attention computation (Shaw et al., 2018). The goal is always the same: put order into an attention computation that has none by itself.',
      ],
      status: 'ready',
    },

    attention: {
      title: 'Consulting each other: attention',
      shortTitle: 'Attention',
      tagline:
        'Each word looks around at the other words in the sentence and pulls in more information from the ones that matter to its meaning.',
      analogy:
        'In the meeting, every participant holds one question (the Query). They compare it with the other participants\' name tags (the Keys), and the better the match, the more of that person\'s material (the Value) they take. The blend of what they collected becomes the new "me".',
      executive: [
        'In "I went to the bank to withdraw money", the word "bank" looks at "money" and settles on the financial meaning. In "We sat on the bank of the river", it looks at "river" and the meaning changes. Attention is this looking-around, implemented with numbers.',
        'Every token goes through this process at the same time. As a result, each token\'s list of numbers shifts from "its meaning on its own" to "its meaning in context".',
        'One point worth remembering from a management perspective: because every token looks at every other token, the computation grows with the square of the sentence length. That is why cost rises quickly as documents get longer.',
      ],
      engineer: [
        "Each token vector x is multiplied by three learned matrices to produce a query Q = xW_Q, a key K = xW_K, and a value V = xW_V. The score is the dot product of Q and K, divided by √d_k and passed through softmax to obtain weights between 0 and 1. Finally V is summed using those weights. In the paper's notation, Attention(Q, K, V) = softmax(QKᵀ / √d_k) V.",
        'The division by √d_k exists because dot products grow with the dimension, which would push softmax into regions where it is nearly one-sided and its gradients become extremely small.',
        "Decoder-only models must not peek at future words, so the scores of tokens that come later are set to -∞. This is the causal mask. The same computation is implemented at a small scale in this repository's src/lib/math.ts.",
      ],
      keyTerms: [
        {
          term: 'Query / Key / Value',
          meaning:
            'Three vectors per token expressing what it is looking for, what it can be found by, and what it passes on.',
        },
        {
          term: 'softmax',
          meaning:
            'A function that turns a set of scores into proportions (probabilities) that sum to 1. Large scores become larger, small scores become smaller.',
        },
      ],
      status: 'ready',
    },

    multihead: {
      title: 'Several viewpoints at once: multi-head attention',
      shortTitle: 'Multi-head attention',
      tagline:
        'Running several attentions side by side lets the model watch different kinds of relationships, such as grammar and meaning, at the same time.',
      analogy:
        'It is like splitting a large meeting into small groups that each discuss a different topic, then bringing the results back together.',
      executive: [
        'A single attention connects words by one criterion only. Real language carries several kinds of relationships at once: who did what, which noun a pronoun refers to, which words form a phrase.',
        'So the Transformer runs several attentions in parallel, called heads, and lets each one learn a different relationship. The outputs of the heads are joined together and passed on to the next step.',
      ],
      engineer: [
        'Each head computes attention independently in a dimension smaller than d_model. The original paper split d_model = 512 into h = 8 heads, giving each head d_k = d_v = 64. The head outputs are concatenated and multiplied by an output matrix W_O to return to d_model dimensions.',
        'Because each head works in a reduced dimension, the total cost is similar to a single head running at full dimensionality. It is a design that gains expressiveness without adding cost.',
      ],
      status: 'ready',
    },

    ffn: {
      title: 'Thinking it over alone: the feed-forward layer',
      shortTitle: 'Feed-forward layer',
      tagline:
        'After attention gathers information, each token digests it on its own, without looking at any other token.',
      analogy:
        'When the meeting ends, everyone returns to their desk to sort out what they heard. Nobody talks to their neighbour during this step.',
      executive: [
        'If attention decides what to take from other words, the feed-forward layer is where each token computes on its own using what it collected. The same computation is applied to every token.',
        "This layer accounts for a large share of the model's parameters. A useful way to remember it: attention handles connection, the feed-forward layer handles processing.",
      ],
      engineer: [
        "It is a two-layer neural network applied identically at every position. A linear transformation widens the dimension (512 → 2048 in the original paper), a non-linear activation is applied (ReLU in the original paper), and a second linear transformation brings it back to d_model. FFN(x) = max(0, xW_1 + b_1)W_2 + b_2. Using the original paper's sizes, the feed-forward parameters in one layer (2 × 512 × 2048 ≈ 2.1 million) are about twice those of the attention projection matrices (4 × 512 × 512 ≈ 1.05 million). GPT-2 uses the GELU activation instead of ReLU (as seen in the published model code).",
        'Information moves between tokens only in attention; the feed-forward layer is applied to each token independently, which makes it easy to parallelize.',
      ],
      status: 'ready',
    },

    stack: {
      title: 'Getting deeper by repeating: stacking layers',
      shortTitle: 'Stacking layers',
      tagline:
        'One attention plus one feed-forward step forms a "layer". Stacking many layers reaches progressively deeper understanding.',
      analogy:
        'It is like refining a document through several rounds of review, while attaching the original at every round so that nothing gets lost.',
      executive: [
        'After one layer, each token reflects a little of its surrounding context. With every additional layer, wider context and more abstract relationships are folded in. Larger models have more layers.',
        "Stacking many layers can blur information. So each layer's result is added back onto its input, a residual connection, and the magnitudes are kept steady with normalization. These two devices make it possible to train deep stacks stably.",
      ],
      engineer: [
        'In the original paper each sub-layer outputs LayerNorm(x + Sublayer(x)), and both the encoder and the decoder stack N = 6 layers. GPT-2 moves the normalization to the input of each sub-layer, the pre-LN arrangement (as seen in the published model code).',
        'Residual connections were proposed to make very deep networks easier to train (He et al., 2016), and the original paper applies them to both sub-layers. Because the input skips the computation and is added back unchanged, the original information survives even in deep stacks.',
      ],
      status: 'ready',
    },

    output: {
      title: 'Choosing the next word: probabilities and temperature',
      shortTitle: 'Probabilities and temperature',
      tagline:
        "The last layer's result is turned into a score for every word in the vocabulary, and the next token is drawn according to those probabilities.",
      analogy:
        'When the meeting ends, the room votes on the next word. Temperature is the dial that sets how strictly the vote is followed: low means only the winner is picked, high means the runners-up get a chance too.',
      executive: [
        "The last token's list of numbers is converted into a score for every token in the vocabulary, and the scores are turned into probabilities that sum to 1. One token is drawn, appended to the text, and the whole process repeats.",
        'Lowering the "temperature" makes the model pick only the most likely token, so answers are stable and repetitive. Raising it gives more variety but also more odd answers. The same model behaves quite differently depending on this setting.',
      ],
      engineer: [
        "The final hidden vector is multiplied by a linear layer to produce vocabulary-sized logits, and softmax turns them into probabilities. The original paper and GPT-2 share this layer's weights with the input embedding.",
        'Temperature T is applied as softmax(z / T). T < 1 sharpens the distribution and T > 1 flattens it. It is commonly combined with top-k sampling, which keeps only the k most likely tokens (Fan et al., 2018), or top-p sampling, which keeps tokens up to a cumulative probability p (Holtzman et al., 2020).',
      ],
      status: 'ready',
    },

    training: {
      title: 'How it learns: training',
      shortTitle: 'Training',
      tagline:
        'Nobody writes the rules. The model plays "guess the next word" over an enormous amount of text, nudging its internal numbers a little each time.',
      analogy:
        'Think of a student working through a workbook with the answers printed in it, billions of times. The question is always "what is the next word?", and the answer is already written in the real text. Check, correct by the size of the mistake, move on.',
      executive: [
        'The training data is a vast amount of text, collected from the web and other sources. The model reads the beginning, predicts the next word, compares it with the real next word, measures the error, and adjusts its parameters a tiny amount in the direction that reduces it. This repeats an enormous number of times.',
        'A model that has finished this "pre-training" is good at continuing text but does not yet answer in a conversational way. So it is refined once more with human-written dialogue examples and human preference ratings. Conversational AI is the product of these two stages.',
        'From a cost perspective, three things jointly determine performance and cost: the amount and quality of data, the model size, and the compute (GPUs).',
      ],
      engineer: [
        'The loss is the cross-entropy of the next token. Backpropagation computes the gradient for every parameter, and an Adam-family optimizer applies the update. The original paper also used Adam.',
        'The pipeline demonstrated by InstructGPT (Ouyang et al., 2022) is pre-training → supervised fine-tuning (SFT) on human-written demonstrations → reinforcement learning from human preference rankings (RLHF). Each stage needs data of a different character.',
      ],
      status: 'ready',
    },

    summary: {
      title: 'Recap: a one-minute summary and next steps',
      shortTitle: 'Recap',
      tagline:
        'A Transformer repeats a "meeting where words consult each other" over several layers to predict the next word.',
      analogy:
        'If you had one sentence in a board meeting: "It is a next-word predictor that reads context on its own, and it keeps improving as you add data and compute."',
      executive: [
        'The card below condenses the whole page into a one-minute read. It can be copied into meeting material as it is.',
      ],
      engineer: [
        'Every link in the "next steps" list below is a verified source, and the original paper is listed with section numbers.',
      ],
      status: 'ready',
    },
  },
}
