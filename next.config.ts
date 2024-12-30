export default {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'avatars.githubusercontent.com'
      },
      {
        protocol: 'https',
        hostname: '*.public.blob.vercel-storage.com'
      }
    ]
  },
  async redirects() {
    return [
      {
        source: '/',
        destination: '/experiments',
        permanent: true
      }
    ];
  }
};
