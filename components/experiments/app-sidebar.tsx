import * as React from 'react';
import { GalleryVerticalEnd } from 'lucide-react';
import experimentsData from '@/experiments.json';
import { useRouter } from 'next/navigation';

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarRail
} from '@/components/ui/sidebar';

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const router = useRouter();

  return (
    <Sidebar {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <a href="#">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                  <GalleryVerticalEnd className="size-4" />
                </div>
                <div className="flex flex-col gap-0.5 leading-none">
                  <span className="font-semibold">Experiments</span>
                </div>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarMenu>
            {experimentsData.map((dataset) => (
              <SidebarMenuItem key={dataset.name}>
                <SidebarMenuButton asChild size="sm">
                  <a href="#">{dataset.name.toUpperCase()}</a>
                </SidebarMenuButton>
                {dataset.splits?.length ? (
                  <SidebarMenuSub>
                    {dataset.splits.map((split) => (
                      <SidebarMenuSubItem key={split.name}>
                        <SidebarMenuSubButton asChild size="sm">
                          <a href="#">{split.name.toUpperCase()}</a>
                        </SidebarMenuSubButton>
                        {split.setups?.length ? (
                          <SidebarMenuSub>
                            {split.setups.map((setup) => {
                              const shortSetup = setup.name
                                .split('_')
                                .map((s) => s[0])
                                .join('')
                                .toUpperCase();
                              return (
                                <SidebarMenuSubItem key={setup.name}>
                                  <SidebarMenuSubButton asChild size="sm">
                                    <a href="#">{shortSetup}</a>
                                  </SidebarMenuSubButton>
                                  {setup.topKs?.length ? (
                                    <SidebarMenuSub>
                                      {setup.topKs.map((topK) => (
                                        <SidebarMenuSubItem key={topK.name}>
                                          <SidebarMenuSubButton
                                            asChild
                                            size="sm"
                                          >
                                            <a href="#">
                                              {topK.name.toUpperCase()}
                                            </a>
                                          </SidebarMenuSubButton>
                                          {topK.models?.length ? (
                                            <SidebarMenuSub>
                                              {topK.models.map((model) => (
                                                <SidebarMenuSubItem
                                                  key={model.name}
                                                >
                                                  <SidebarMenuSubButton
                                                    asChild
                                                    size="sm"
                                                  >
                                                    <a
                                                      href="#"
                                                      onClick={(e) => {
                                                        e.preventDefault();
                                                        router.push(
                                                          `?model=${encodeURIComponent(
                                                            model.path
                                                          )}`
                                                        );
                                                      }}
                                                    >
                                                      {model.name}
                                                    </a>
                                                  </SidebarMenuSubButton>
                                                </SidebarMenuSubItem>
                                              ))}
                                            </SidebarMenuSub>
                                          ) : null}
                                        </SidebarMenuSubItem>
                                      ))}
                                    </SidebarMenuSub>
                                  ) : null}
                                </SidebarMenuSubItem>
                              );
                            })}
                          </SidebarMenuSub>
                        ) : null}
                      </SidebarMenuSubItem>
                    ))}
                  </SidebarMenuSub>
                ) : null}
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarGroup>
      </SidebarContent>
      <SidebarRail />
    </Sidebar>
  );
}
