import { IDBPDatabase, openDB } from 'idb';
import { createContext, useContext, useEffect, useState } from 'react';

export const DB_NAME = 'AnnotationDB';
export const STORE_NAME = 'currentAnnotations';

const useIDB = () => {
  const [db, setDb] = useState<IDBPDatabase | null>(null);

  useEffect(() => {
    openDB(DB_NAME, 1, {
      upgrade(db) {
        db.createObjectStore(STORE_NAME, {
          keyPath: 'id',
          autoIncrement: true
        });
      }
    }).then((db) => {
      setDb(db);
    });
  }, []);

  const getCurrentAnnotation = async () => {
    if (!db) {
      return null;
    }
    return db.get(STORE_NAME, 0);
  };

  const updateCurrentAnnotation = async (annotation: any) => {
    if (!db) {
      return null;
    }
    if (annotation.id === undefined) {
      return null;
    }
    return db.put(STORE_NAME, annotation);
  };

  const deleteCurrentAnnotation = async () => {
    if (!db) {
      return null;
    }
    return db.delete(STORE_NAME, 0);
  };

  return {
    db,
    getCurrentAnnotation,
    updateCurrentAnnotation,
    deleteCurrentAnnotation
  };
};

const IDBContext = createContext<any>(null);

export const useIDBContext = () => useContext(IDBContext);
export const IDBProvider = ({ children }: { children: React.ReactNode }) => {
  const idbMethods = useIDB();
  if (!idbMethods.db) {
    return <div>Loading...</div>;
  }
  return (
    <IDBContext.Provider value={idbMethods}>{children}</IDBContext.Provider>
  );
};
