export const DB_NAME = 'AnnotationDB';
export const STORE_NAME = 'currentAnnotations';

export const DBConfig = {
  name: DB_NAME,
  version: 1,
  objectStoresMeta: [
    {
      store: STORE_NAME,
      storeConfig: { keyPath: 'id', autoIncrement: true },
      storeSchema: []
    }
  ]
};
