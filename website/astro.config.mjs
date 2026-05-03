import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://cyclops.gopalji.me',
  integrations: [
    starlight({
      title: 'Cyclops',
      description: 'Minimal Python agent framework. Any LLM. MCP native.',
      favicon: '/favicon.svg',
      defaultLocale: 'en',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/gopaljigaur/cyclops' },
      ],
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Introduction', slug: 'index' },
            { label: 'Getting Started', slug: 'getting-started' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Agents', slug: 'guides/agents' },
            { label: 'Tools', slug: 'guides/tools' },
            { label: 'Streaming', slug: 'guides/streaming' },
            { label: 'Structured Output', slug: 'guides/structured-output' },
            { label: 'Memory', slug: 'guides/memory' },
            { label: 'MCP', slug: 'guides/mcp' },
            { label: 'Plugins', slug: 'guides/plugins' },
            { label: 'Cost Tracking', slug: 'guides/cost-tracking' },
            { label: 'Hooks', slug: 'guides/hooks' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'API Reference', slug: 'api-reference' },
          ],
        },
      ],
      customCss: ['./src/styles/custom.css'],
      editLink: {
        baseUrl: 'https://github.com/gopaljigaur/cyclops/edit/main/website/',
      },
    }),
  ],
});
