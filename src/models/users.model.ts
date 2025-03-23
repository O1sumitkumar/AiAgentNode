import { model, Schema, Document } from 'mongoose';
import { User } from '@interfaces/users.interface';

const UserSchema: Schema = new Schema(
  {
    email: {
      type: String,
      required: true,
      unique: true,
      trim: true,
      lowercase: true,
      validate: {
        validator: function (v: string) {
          return /^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/.test(v);
        },
        message: props => `${props.value} is not a valid email address!`,
      },
      index: true, // adds database index for better query performance
    },
    password: {
      type: String,
      required: true,
      minlength: 6,
    },
    role: {
      type: String,
      required: true,
      default: 'user',
      enum: ['user', 'admin'], // restrict to valid roles
    },
    createdAt: {
      type: Date,
      default: Date.now,
      immutable: true, // prevents modification after creation
    },
    updatedAt: {
      type: Date,
      default: Date.now,
    },
  },
  {
    timestamps: true, // automatically handle createdAt and updatedAt
    collection: 'users', // explicitly name collection
    versionKey: false, // remove __v field
  },
);

export const UserModel = model<User & Document>('User', UserSchema);
