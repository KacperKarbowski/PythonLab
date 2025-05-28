from django.core.management.base import BaseCommand
from RybkiApp.models import Post
import os

class Command(BaseCommand):
    help = 'Usuwa wszystkie posty i powiązane obrazki'

    def handle(self, *args, **kwargs):
        self.stdout.write("🧹 Usuwam wszystkie posty...")

        for post in Post.objects.all():
            if post.image and os.path.isfile(post.image.path):
                os.remove(post.image.path)
                self.stdout.write(f"🗑 Usunięto obrazek: {post.image.path}")
            post.delete()

        self.stdout.write(self.style.SUCCESS('✅ Wszystkie posty i obrazki usunięte.'))
