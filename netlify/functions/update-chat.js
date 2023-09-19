const { PrismaClient } = require("@prisma/client");
const OpenAI = require("openai");
const { ulid } = require("ulid");

const prisma = new PrismaClient();
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const getChatTitle = async (messages) => {
  const llmResult = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    temperature: 0.1,
    messages: [
      {
        role: "system",
        content: "Summarize the following into a very short title",
      },
      {
        role: "user",
        content: JSON.stringify(messages).substring(0, 5000),
      },
    ],
  });
  return llmResult.choices[0].message.content;
};

exports.handler = async (event, context) => {
  const { apiKey, id, messages, doDelete } = JSON.parse(event.body);
  if (!apiKey) return { statusCode: 400, body: "No API key provided" };
  console.log(JSON.parse(event.body));
  const user = await prisma.user.findFirst({
    where: { apiKey: apiKey },
  });
  let chat;
  if (id === null) {
    chat = await prisma.chat.create({
      data: {
        id: ulid(),
        name: await getChatTitle(messages),
        userId: user.id,
      },
    });
    await prisma.message.createMany({
      data: messages.map((message) => ({
        role: message.role,
        text: message.content,
        chatId: chat.id,
      })),
    });
  } else {
    chat = await prisma.chat.findFirst({
      where: { id },
    });
    await prisma.message.deleteMany({
      where: { chatId: id },
    });
    if (doDelete) {
      await prisma.chat.delete({
        where: { id },
      });
    } else {
      await prisma.message.createMany({
        data: messages.map((message) => ({
          role: message.role,
          text: message.content,
          chatId: chat.id,
        })),
      });
    }
  }
  return {
    statusCode: 200,
    body: JSON.stringify(chat, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value
    ),
  };
};