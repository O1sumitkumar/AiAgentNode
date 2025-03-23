import { connect, set } from 'mongoose';
import { NODE_ENV, DB_HOST, DB_PORT, DB_DATABASE } from '@config';

export const dbConnection = async () => {
  const dbConfig = {
    url: `mongodb+srv://${DB_HOST}:${DB_PORT}/${DB_DATABASE}`,
    options: {
      serverSelectionTimeoutMS: 5000, // Wait up to 5 seconds for a server to be available
      socketTimeoutMS: 45000, // Wait up to 45 seconds for a response
      autoIndex: true, // Automatically build indexes (set to false in production)
    },
  };

  if (NODE_ENV !== 'production') {
    set('debug', true);
  }
  console.log(dbConfig.url);

  try {
    await connect(dbConfig.url, dbConfig.options);
    console.log('Database connected successfully!');
  } catch (error) {
    console.error('Error connecting to the database:', error);
    // Optionally, exit the process if the connection fails
    process.exit(1);
  }
};
