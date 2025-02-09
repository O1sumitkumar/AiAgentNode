import { connect, set } from 'mongoose';
import { NODE_ENV, DB_HOST, DB_PORT, DB_DATABASE } from '@config';

export const dbConnection = async () => {
  const dbConfig = {
    url: `mongodb://${DB_HOST}:${DB_PORT}/${DB_DATABASE}`,
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

  await connect(dbConfig.url, dbConfig.options);
};
